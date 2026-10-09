"""FastAPI routes for OpenAI-compatible transcription."""
from __future__ import annotations

import asyncio
import bisect
import functools
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response

from . import aligner, retime, spoken
from .audio import load_audio
from .chunker import plan_chunks, slice_chunks
from .config import (
    ALIGNER_CONFIGS,
    CHUNK_CONTEXT_SEC,
    CHUNK_MIN_SEC,
    COMPARE_UI,
    CPU_INFO,
    LANGUAGE_CODE,
    MAX_AUDIO_SECONDS,
    MAX_BATCH_BYTES,
    MAX_BATCH_FILES,
    MAX_REQUEST_CHUNKS,
    MAX_UPLOAD_BYTES,
    MODEL_CONFIGS,
    RETIME_WORDS,
    SPOKEN_NUMBERS,
    TARGET_SR,
    UPLOAD_READ_CHUNK_BYTES,
    logger,
)
from .model import ModelLoadError, loaded_models, runtime_status, variant_key
from .schemas import (
    AlignerCard,
    AlignerList,
    BatchTranscription,
    ErrorResponse,
    Health,
    ModelCard,
    ModelList,
    Ready,
    Transcription,
    VerboseTranscription,
)

# Name operationIds, and the docs' request body schemas, after the handler
# (`transcribe`, `Body_transcribe`) rather than handler, path and method.
router = APIRouter(generate_unique_id_function=lambda route: route.name)
_ALLOWED_FORMATS = ("json", "text", "srt", "vtt", "verbose_json")  # in the docs' order
_GRANULARITIES = {"word", "segment"}  # OpenAI's; segments come with verbose_json anyway

# Parakeet TDT reports token START times only (80 ms encoder frames); its
# duration head emits at most 4 frames per token, so a token's audio never
# extends more than 0.32 s past its start.
_WORD_TAIL_SEC = 0.32


@dataclass(slots=True)
class _PreparedAudio:
    waveform: Any
    ranges: List[Tuple[int, int]]  # where each piece's words come from
    windows: List[Tuple[int, int]]  # what each piece decodes: its range and context
    speech: List[Tuple[int, int]]  # what VAD heard as speech (none for a short clip)
    pieces: List[Any]  # the windows' audio
    duration: float


class _AudioTooLong(ValueError):
    pass


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u2581", " ").strip()
    text = re.sub(r"\s+", " ", text)
    # onnx_asr drops the space before any non-word character, meaning to join
    # punctuation, and so glues currency to the word before: "was\u00a31.10".
    text = re.sub(r"(?<=[^\W\d_])(?=[$\u00a3\u20ac])", " ", text)
    return text.replace(" '", "'")


def _fmt_srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _segments_to_srt(segments: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    index = 1
    for segment in segments:
        text = segment["segment"].strip()
        if not text:
            continue
        lines.extend(
            [
                str(index),
                f"{_fmt_srt_time(segment['start'])} --> {_fmt_srt_time(segment['end'])}",
                text,
                "",
            ]
        )
        index += 1
    return "\n".join(lines)


def _segments_to_vtt(segments: Sequence[Dict[str, Any]]) -> str:
    output = ["WEBVTT", ""]
    for segment in segments:
        text = segment["segment"].strip()
        if not text:
            continue
        start = _fmt_srt_time(segment["start"]).replace(",", ".")
        end = _fmt_srt_time(segment["end"]).replace(",", ".")
        output.extend([f"{start} --> {end}", text, ""])
    return "\n".join(output)


def _extract(result: Any) -> Dict[str, Any]:
    text = _clean_text(getattr(result, "text", str(result)))
    tokens = [str(token) for token in (getattr(result, "tokens", []) or [])]
    raw_timestamps = list(getattr(result, "timestamps", []) or [])
    if tokens and len(raw_timestamps) != len(tokens):
        logger.warning(
            "token/timestamp length mismatch: %d tokens vs %d timestamps",
            len(tokens),
            len(raw_timestamps),
        )
    # Substitute the previous timestamp for missing/invalid entries instead of
    # dropping them, so tokens and timestamps always stay aligned 1:1.
    timestamps: List[float] = []
    previous = 0.0
    for index in range(len(tokens)):
        try:
            timestamp = float(raw_timestamps[index])
        except (IndexError, TypeError, ValueError):
            timestamp = previous
        if not math.isfinite(timestamp) or timestamp < 0.0:
            timestamp = previous
        timestamps.append(timestamp)
        previous = timestamp
    return {"text": text, "tokens": tokens, "timestamps": timestamps}


def _validate_model(model: str) -> str:
    normalized = model.strip().lower()
    if normalized not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {model!r}. Available models: {sorted(MODEL_CONFIGS)}",
        )
    return normalized


def _named(value: str, field: str, quantization: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """A `model` or `aligner` value, "name" or "name:quantization", as (name,
    quantization). A model's quantization may instead come from the
    `quantization` field, and must agree when both are sent."""
    name, colon, suffix = value.partition(":")
    if not colon:
        return value, quantization
    suffix = suffix.strip().lower()
    if not suffix:
        raise HTTPException(status_code=400, detail=f"{field} {value!r} has no quantization after ':'")
    if quantization is not None and quantization.strip().lower() != suffix:
        raise HTTPException(
            status_code=400, detail=f"{field} {value!r} says {suffix!r} but quantization says {quantization!r}"
        )
    return name, suffix


def _variant(model_name: str, quantization: Optional[str]) -> str:
    """The "model:quant" key a request runs on; fp32 unless it asks otherwise."""
    try:
        return variant_key(model_name, quantization)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


def _family(model_name: str) -> str:
    return MODEL_CONFIGS[model_name]["family"]


def _chunk_bounds(model_name: str) -> Tuple[float, float, float, float]:
    """(target, max, min, context) seconds for chunking this model (models.yaml).
    Context (PARAKEET_CHUNK_CONTEXT_SEC) is Parakeet's only: Whisper returns no
    word times to trim it back out by."""
    config = MODEL_CONFIGS[model_name]
    target, maximum = config["chunk_target_sec"], config["chunk_max_sec"]
    context = min(CHUNK_CONTEXT_SEC, maximum / 4) if config["family"] == "parakeet" else 0.0
    return target, maximum, min(CHUNK_MIN_SEC, target), context


def _validate_format(response_format: str) -> str:
    normalized = (response_format or "json").strip().lower()
    if normalized not in _ALLOWED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format {response_format!r}",
        )
    return normalized


async def _read_upload_limited(upload: UploadFile) -> bytes:
    if not upload or not upload.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    declared_size = getattr(upload, "size", None)
    if declared_size is not None and declared_size > MAX_UPLOAD_BYTES:
        await upload.close()
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES} byte upload limit",
        )

    payload = bytearray()
    try:
        while True:
            chunk = await upload.read(UPLOAD_READ_CHUNK_BYTES)
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds the {MAX_UPLOAD_BYTES} byte upload limit",
                )
    finally:
        await upload.close()
    if not payload:
        raise HTTPException(status_code=400, detail="Empty file")
    return bytes(payload)


def _prepare_audio(
    raw: bytes,
    target_sec: float,
    max_sec: float,
    min_sec: float,
    context_sec: float = 0.0,
) -> _PreparedAudio:
    waveform = load_audio(raw)
    duration = float(waveform.size) / TARGET_SR
    if duration <= 0:
        raise ValueError("decoded audio is empty")
    if duration > MAX_AUDIO_SECONDS:
        raise _AudioTooLong(
            f"audio duration {duration:.1f}s exceeds limit {MAX_AUDIO_SECONDS:.1f}s"
        )
    plan = plan_chunks(
        waveform, target_sec=target_sec, max_sec=max_sec, min_sec=min_sec, context_sec=context_sec
    )
    pieces = slice_chunks(waveform, plan.windows)
    if len(pieces) > MAX_REQUEST_CHUNKS:
        raise _AudioTooLong(
            f"audio produced {len(pieces)} chunks; limit is {MAX_REQUEST_CHUNKS}"
        )
    return _PreparedAudio(
        waveform=waveform,
        ranges=plan.ranges,
        windows=plan.windows,
        speech=plan.speech,
        pieces=pieces,
        duration=duration,
    )


async def _prepare_in_pool(
    request: Request,
    raw: bytes,
    target_sec: float,
    max_sec: float,
    min_sec: float,
    context_sec: float = 0.0,
) -> _PreparedAudio:
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            request.app.state.audio_pool,
            _prepare_audio,
            raw,
            target_sec,
            max_sec,
            min_sec,
            context_sec,
        )
    except _AudioTooLong as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("audio decode/preprocessing failed")
        raise HTTPException(status_code=415, detail="Audio could not be decoded") from exc


def _apply_alignment(
    words: List[Dict[str, Any]],
    spans: Optional[Sequence[Optional[aligner.Span]]],
    chunk_start: float,
    chunk_end: float,
) -> None:
    """Re-time one chunk's words from aligner spans (seconds from chunk start).

    Words the aligner could not place keep their model times, squeezed between
    their aligned neighbours so the word list stays in order.
    """
    if not spans:
        return
    for word, span in zip(words, spans):
        if span is not None:
            word["start"] = min(chunk_end, chunk_start + span[0])
            word["end"] = min(chunk_end, max(word["start"], chunk_start + span[1]))
    for index, (word, span) in enumerate(zip(words, spans)):
        if span is not None:
            continue
        low = words[index - 1]["end"] if index else chunk_start
        high = next(
            (w["start"] for w, s in zip(words[index + 1 :], spans[index + 1 :]) if s is not None),
            chunk_end,
        )
        high = max(low, high)
        word["start"] = min(max(word["start"], low), high)
        word["end"] = min(max(word["end"], word["start"]), high)


# Parakeet's number is trusted unless the audio prefers a mishearing of it
# (spoken.rivals) by at least this much, in log-prob over the phrase's window.
# Measured on the TTS benchmarks (381 clips of alternative readings, 306-clip
# corpus) and on spoken lists of 1-99 (571 numbers): no mishearing beat a
# right number by more than 11.7, and at 10 some did ("three nil" to "two
# nil"); the smallest real fix led by 16.3. At 20, 9 + 3 misheard numbers are
# fixed and none broken; 15 fixes one more, 40 and above lose most of them.
_CORRECTION_MARGIN = 20.0
_ZERO = re.compile(rf"(\W*)({'|'.join(spoken.ZERO_WORDS)})(\W*)")


def _phrases(texts: Sequence[str], most: Optional[int] = None) -> List[spoken.Phrase]:
    """The numbers to say in one chunk's words (spoken.phrases): none in a
    chunk mostly in another alphabet (Cyrillic, Greek, ...), which is not
    English whatever the request's `language` said; the aligner leaves those
    untimed by the same rule. Checked second: a chunk with no digit has no
    phrase, and that is the cheaper test."""
    found = spoken.phrases(texts, most=most)
    return [] if found and aligner.other_alphabet(texts) else found


def _undecided(texts: Sequence[str], phrase: spoken.Phrase) -> bool:
    """Whether the audio has anything to decide for `phrase`: which of its
    readings was said, or whether a misheard number was."""
    return len(phrase.options) > 1 or bool(spoken.rivals(texts, phrase))


def _choose_readings(
    texts: List[str], found: Sequence[spoken.Phrase], chunk: aligner.ChunkAligner
) -> List[str]:
    """Pick, by ear, how each phrase was said, or which number was.

    The default readings are aligned first to find each phrase's neighbours;
    each phrase is then heard from the end of the word before it to the start
    of the word after it, so a wrong default only costs its own slot.
    """
    # ponytail: the wav2vec2 pass covers the whole chunk (~4 s of CPU per 60 s)
    # even to hear one number (~0.3 s for its window plus _CONTEXT). For
    # requests without word times, run it over each phrase's window from
    # Parakeet's own times instead, if the cost matters.
    chosen = [phrase.options[0] for phrase in found]
    said = list(texts)  # spoken.spoken_words(texts), without finding the phrases again
    for phrase in found:
        said[phrase.start : phrase.end] = phrase.readings[0]
    spans = chunk.spans(said) or [None] * len(texts)
    for index, phrase in enumerate(found):
        rivals = spoken.rivals(texts, phrase)
        if len(phrase.options) < 2 and not rivals:
            continue
        start = next((s[1] for s in reversed(spans[: phrase.start]) if s is not None), 0.0)
        end = next((s[0] for s in spans[phrase.end :] if s is not None), float("inf"))
        own = chunk.scores(phrase.options, start, end)
        choice = phrase.options[own.index(max(own))]
        if rivals and max(own) > float("-inf"):
            theirs = chunk.scores(rivals, start, end)
            if max(theirs) > max(own) + _CORRECTION_MARGIN:
                choice = rivals[theirs.index(max(theirs))]
        chosen[index] = _each_zero(choice, chunk, start, end)
    return chosen


def _each_zero(said: str, chunk: aligner.ChunkAligner, start: float, end: float) -> str:
    """Speakers mix their zeros within one number ("nine zero oh one"): try
    each zero the other ways too, one at a time, punctuation kept. The zero
    already said comes first, so it stays when the audio can't tell."""
    tokens = said.split()
    for slot, token in enumerate(tokens):
        if match := _ZERO.fullmatch(token):
            opening, said_zero, closing = match.groups()
            zeros = [said_zero, *(zero for zero in spoken.ZERO_WORDS if zero != said_zero)]
            ways = [[*tokens[:slot], f"{opening}{zero}{closing}", *tokens[slot + 1 :]] for zero in zeros]
            tokens = ways[chunk.best([" ".join(way) for way in ways], start, end)]
    return " ".join(tokens)


def _speak_numbers(
    words: List[Dict[str, Any]], chunk: Callable[[], Optional[aligner.ChunkAligner]]
) -> Optional[List[Dict[str, Any]]]:
    """One chunk's words with numbers, money and units said out ("$5" -> "five
    dollars"), or None if it has none (PARAKEET_SPOKEN_NUMBERS).

    Each number is a phrase with the words that change how it is said
    (spoken.phrases). The audio, via `chunk()` (called only when a phrase has
    something to decide, as it loads the aligner), picks how it was said
    ("£2.10": "two pounds ten", "two pounds and ten pence", ...) and may, by a
    clear margin, correct a misheard number; without it the first reading is
    used. A phrase shares its time span out by length among its spoken words;
    when the aligner runs next it re-times each of them from the audio.
    """
    texts = [w["word"] for w in words]
    found = _phrases(texts, most=2)  # every reading only if the audio will choose
    if not found:
        return None
    chosen = [phrase.options[0] for phrase in found]
    if any(_undecided(texts, phrase) for phrase in found) and (ear := chunk()) is not None:
        chosen = _choose_readings(texts, _phrases(texts), ear)
    out: List[Dict[str, Any]] = []
    done = 0
    for phrase, text in zip(found, chosen):
        out += words[done : phrase.start]
        if spoken.starts_sentence(texts[phrase.start - 1] if phrase.start else None):
            text = spoken.capitalize(text)
        out += _spread(text.split(), words[phrase.start]["start"], words[phrase.end - 1]["end"])
        done = phrase.end
    return out + words[done:]


def _spread(parts: List[str], start: float, end: float) -> List[Dict[str, Any]]:
    """Words over the span [start, end], each a share by its length."""
    share = (end - start) / sum(len(part) for part in parts)
    out = []
    for position, part in enumerate(parts):
        stop = end if position == len(parts) - 1 else start + share * len(part)
        out.append({"start": start, "end": stop, "word": part})
        start = stop
    return out


def _word_spans(tokens: Sequence[str]) -> List[Tuple[str, int, int]]:
    """A result's BPE pieces as (word, first, last): the indices of the pieces
    each word starts and ends with.

    A piece starting with the word marker ("\u2581" or a plain space, depending
    on export) opens a new word. Parakeet emits the marker as a token of its
    own before digits and currency signs (" was", " ", "\u00a3", "1"...): it
    opens the next piece's word.
    """
    spans: List[Tuple[str, int, int]] = []
    pending_break = False
    for index, token in enumerate(tokens):
        piece = token.replace("\u2581", " ")
        starts_word = pending_break or piece.startswith(" ")
        piece = piece.strip()
        pending_break = starts_word and not piece
        if not piece:
            continue
        if spans and not starts_word:
            word, first, _last = spans[-1]
            spans[-1] = (word + piece, first, index)
        else:
            spans.append((piece, index, index))
    return spans


def _group_words(info: Dict[str, Any]) -> List[Tuple[str, float, float]]:
    """One chunk's BPE pieces (from _extract) as (word, first_ts, last_ts)."""
    timestamps = info["timestamps"]
    return [(word, timestamps[first], timestamps[last]) for word, first, last in _word_spans(info["tokens"])]


# How onnx_asr joins a result's tokens into its text: the leading space, and
# any space before a non-word character, go.
_DECODE_SPACE = re.compile(r"\A\s|\s\B|(\s)\b")


def _trimmed(prepared: _PreparedAudio, results: Sequence[Any]) -> List[Any]:
    """Each piece's result cut back to its own range: the words that start in
    it, timed from its start, with its text rebuilt from their tokens as
    onnx_asr builds it. The context either side (plan_chunks) is decoded only
    so that the piece's edge words are not ones Parakeet makes up as its input
    ends (#68); the words in it are the neighbours'. A piece decoded without
    context comes back as it was."""
    out: List[Any] = []
    for (start, end), (window_start, window_end), result in zip(prepared.ranges, prepared.windows, results):
        info = None if (start, end) == (window_start, window_end) else _extract(result)
        if not (info and info["tokens"]):
            out.append(result)  # Whisper (no tokens) never gets context: _chunk_bounds
            continue
        tokens, timestamps = info["tokens"], info["timestamps"]
        head = (start - window_start) / TARGET_SR
        tail = (end - window_start) / TARGET_SR if window_end > end else math.inf
        kept: List[int] = []
        previous = -1
        for _word, first, last in _word_spans(tokens):
            if head <= timestamps[first] < tail:
                kept.extend(range(previous + 1, last + 1))  # with the lone markers before it
            previous = last
        text = "".join(tokens[index].replace("\u2581", " ") for index in kept)
        out.append(
            SimpleNamespace(
                text=_DECODE_SPACE.sub(lambda match: " " if match.group(1) else "", text),
                tokens=[tokens[index] for index in kept],
                timestamps=[max(0.0, timestamps[index] - head) for index in kept],
            )
        )
    return out


def _stitch(
    prepared: _PreparedAudio,
    results: Sequence[Any],
    *,
    align: bool = False,
    speak: bool = False,
    language: Optional[str] = None,
    aligner_choice: Optional[Tuple[str, str]] = None,
    retime_words: bool = False,
    kept: Optional[Sequence[Any]] = None,
) -> Tuple[str, List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """Chunk results -> (text, segments, words); words None when a text-only
    (Whisper) chunk was to be aligned and could not be. `kept` is the results
    already cut to their ranges (_trimmed), if the caller has them.

    `align` re-times words with the request's aligner (`aligner_choice`, its
    name and quantization); `speak` says numbers out (PARAKEET_SPOKEN_NUMBERS),
    hearing which reading was said through that aligner if there is one. Either
    may run the aligner's model: call it through _stitch_request, which keeps
    that off the event loop. `retime_words` moves the words of each chunk the
    aligner did not re-time out of the pauses they slipped into (retime.py).
    """
    if len(results) != len(prepared.ranges):
        raise RuntimeError(
            f"inference returned {len(results)} results for "
            f"{len(prepared.ranges)} chunks"
        )
    results = _trimmed(prepared, results) if kept is None else kept

    segments: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []
    untimed = False
    gaps = gap_ends = None
    if retime_words and prepared.waveform is not None:
        gaps = retime.pauses(prepared.waveform)
        gap_ends = [end for _start, end in gaps]
    for (start_sample, end_sample), window, piece, result in zip(
        prepared.ranges, prepared.windows, prepared.pieces, results
    ):
        # The aligner hears the range's own audio, as its words were trimmed to.
        chunk_wav = piece
        if window != (start_sample, end_sample):
            chunk_wav = piece[start_sample - window[0] : end_sample - window[0]]
        chunk_start = start_sample / TARGET_SR
        chunk_end = min(prepared.duration, end_sample / TARGET_SR)
        info = _extract(result)
        if not info["text"]:
            continue

        timestamps = info["timestamps"]
        segment_start = chunk_start
        if timestamps:
            segment_start = min(chunk_end, max(chunk_start, chunk_start + timestamps[0]))
        segment_end = max(segment_start, chunk_end)
        if timestamps:
            segment_end = min(
                segment_end, max(segment_start, chunk_start + timestamps[-1] + _WORD_TAIL_SEC)
            )
        segments.append(
            {
                "start": segment_start,
                "end": segment_end,
                "segment": info["text"],
            }
        )

        grouped = _group_words(info)
        chunk_words: List[Dict[str, Any]] = []
        for index, (word, first_ts, last_ts) in enumerate(grouped):
            word_start = min(chunk_end, max(chunk_start, chunk_start + first_ts))
            if index + 1 < len(grouped):
                next_start = chunk_start + grouped[index + 1][1]
            else:
                next_start = chunk_end
            # The model only reports token start times, so bound the end by the
            # last token's start plus the model's maximum token duration rather
            # than letting a word absorb the silence before the next word.
            word_end = min(next_start, chunk_start + last_ts + _WORD_TAIL_SEC)
            word_end = min(chunk_end, max(word_start, word_end))
            chunk_words.append({"start": word_start, "end": word_end, "word": word})

        heard: List[Optional[aligner.ChunkAligner]] = []

        def chunk(wav: Any = chunk_wav) -> Optional[aligner.ChunkAligner]:
            """The chunk's aligner, made on first use (it loads the model) and shared."""
            if not heard:
                heard.append(aligner.for_chunk(wav, language, *aligner_choice) if aligner_choice else None)
            return heard[0]

        said = None
        if speak:
            try:
                said = _speak_numbers(chunk_words, chunk)
            except Exception:  # it refines a finished transcript: never a 500 for it
                logger.exception("spoken numbers failed; keeping Parakeet's text")
        if said is not None:
            # Text and words are rewritten together so they always agree.
            chunk_words = said
            segments[-1]["segment"] = _clean_text(" ".join(w["word"] for w in chunk_words))

        # A text-only result (Whisper) has no tokens, so no word spans exist to
        # align. When alignment will run, split the transcript into words with
        # placeholder spans for the aligner to re-time from the audio; they are
        # kept only if it does, never returned as word times of their own.
        placeholders = align and not chunk_words and bool(info["text"])
        if placeholders:
            chunk_words = _spread(info["text"].split(), chunk_start, chunk_end)

        spans = None
        if align and chunk_words and (timer := chunk()) is not None:
            spans = timer.spans([w["word"] for w in chunk_words])
        if placeholders and not any(spans or ()):
            untimed = True  # no aligner, no language, or nothing placed
        elif spans:
            _apply_alignment(chunk_words, spans, chunk_start, chunk_end)
            _cover(segments[-1], chunk_words)
        elif gaps is not None and chunk_words and not placeholders:
            chunk_words = retime.retime(
                chunk_words, retime.within(gaps, gap_ends, chunk_start, chunk_end), chunk_start, chunk_end
            )
            _cover(segments[-1], chunk_words)
        words.extend(chunk_words)

    full_text = _clean_text(" ".join(item["segment"] for item in segments))
    # Whisper's word times come only from the aligner: none rather than some.
    return full_text, segments, None if untimed else words


def _cover(segment: Dict[str, Any], words: List[Dict[str, Any]]) -> None:
    """Segment bounds came from the model's estimates; keep them covering its
    re-timed words, so a cue never ends before its last word."""
    segment["start"] = min(segment["start"], words[0]["start"])
    segment["end"] = max(segment["end"], words[-1]["end"])


def _needs_aligner(
    results: Sequence[Any],
    *,
    align: bool = False,
    speak: bool = False,
    aligner_choice: Optional[Tuple[str, str]] = None,
) -> bool:
    """Whether _stitch, with these flags, runs the aligner's model: for word
    times, or to hear how a number was said (a phrase with something to
    decide). The same words and phrases as _stitch's, so they agree; two
    readings of a phrase are enough to know it has a choice."""
    if align:
        return True
    if not (speak and aligner_choice):
        return False  # for_chunk answers None without loading anything
    try:
        for result in results:
            texts = [word for word, _first, _last in _group_words(_extract(result))]
            if any(_undecided(texts, phrase) for phrase in _phrases(texts, most=2)):
                return True
    except Exception:  # can't tell: _stitch will log it and keep Parakeet's text
        return True
    return False


async def _stitch_request(
    request: Request, prepared: _PreparedAudio, results: Sequence[Any], **flags: Any
) -> Tuple[str, List[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    """_stitch(prepared, results, **flags), off the event loop unless it is
    plain text work (no word times, no digit to say, nothing to re-time) on a
    clip's one piece.

    Long audio's pieces are thousands of tokens to trim and join: plain text
    from them, saying numbers (which reads their readings back through
    number_parse), and deciding whether the aligner's model is needed run on
    the audio pool. With the model (a second ONNX model over the audio) it runs
    on the one-worker align pool, off the audio pool so it never holds up other
    requests' decoding; only those requests queue behind other alignment."""
    says_numbers = flags.get("speak") and any(
        char.isdigit() for result in results for char in str(getattr(result, "text", result))
    )
    loop, state = asyncio.get_running_loop(), request.app.state
    if not (flags.get("align") or says_numbers or flags.get("retime_words")):
        stitch = functools.partial(_stitch, prepared, results, **flags)
        return stitch() if len(results) < 2 else await loop.run_in_executor(state.audio_pool, stitch)

    def needs_aligner() -> Tuple[List[Any], bool]:
        """The words _stitch keeps (_trimmed), for it too, and _needs_aligner over them."""
        kept = _trimmed(prepared, results)
        return kept, _needs_aligner(kept, speak=flags.get("speak", False), aligner_choice=flags.get("aligner_choice"))

    kept, needs = None, bool(flags.get("align"))
    if says_numbers and not needs:
        kept, needs = await loop.run_in_executor(state.audio_pool, needs_aligner)
    stitch = functools.partial(_stitch, prepared, results, kept=kept, **flags)
    return await loop.run_in_executor(state.align_pool if needs else state.audio_pool, stitch)


async def _infer(request: Request, pieces: List[Any], model_key: str):
    worker = request.app.state.worker
    if worker is None or not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="Model is not ready")
    try:
        return await worker.submit_many(pieces, model_key)
    except ModelLoadError as exc:  # load_model logged it
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# Parakeet sometimes stops partway through a window, or skips a stretch of
# it, and returns nothing for tens of seconds of speech, depending on exactly
# where the window starts or ends (#69). A piece with this much speech, by
# VAD, between two of its tokens inside its own range has skipped it.
_STALL_SEC = 3.0


def _speech_within(speech: Sequence[Tuple[int, int]], starts: Sequence[int], low: int, high: int) -> int:
    """Samples of VAD speech between `low` and `high` (`starts`: each span's start)."""
    total = 0
    for speech_start, speech_end in speech[max(0, bisect.bisect_right(starts, low) - 1) :]:
        if speech_start >= high:
            break
        total += max(0, min(speech_end, high) - max(speech_start, low))
    return total


def _stalled(prepared: _PreparedAudio, results: Sequence[Any]) -> Dict[int, List[Tuple[int, int]]]:
    """The pieces that decoded context (plan_chunks) and skipped at least
    _STALL_SEC of speech in their own range: before their first token, between
    two, or after their last. Each comes with the stretches it skipped, in
    samples. A piece decoded as just its range is the same audio as before
    context, which has not done this."""
    starts = [start for start, _end in prepared.speech]
    stall, tail = int(_STALL_SEC * TARGET_SR), int(_WORD_TAIL_SEC * TARGET_SR)
    skipped: Dict[int, List[Tuple[int, int]]] = {}
    for index, ((start, end), window, result) in enumerate(zip(prepared.ranges, prepared.windows, results)):
        if window == (start, end):
            continue
        heard = sorted(window[0] + int(at * TARGET_SR) for at in _extract(result)["timestamps"])
        for low, high in zip([window[0]] + [at + tail for at in heard], heard + [window[1]]):
            low, high = max(low, start), min(high, end)
            if high - low >= stall and _speech_within(prepared.speech, starts, low, high) >= stall:
                skipped.setdefault(index, []).append((low, high))
    return skipped


# A piece decoded again without context keeps the redo only if it hears at
# least this many words in the speech the first decode skipped, and no fewer
# words in all. What VAD calls speech may be music, laughter or noise, with no
# words to find (PARAKEET_VAD=volume above all), and without context Parakeet
# may make up a word as its input ends (#68): one new word there is no sign of
# a skip.
_REDO_MIN_WORDS = 2


def _words_in(result: Any, origin: int, stretches: Sequence[Tuple[int, int]]) -> int:
    """How many of `result`'s words, decoded from sample `origin`, start in `stretches`."""
    info = _extract(result)
    starts = [origin + int(info["timestamps"][first] * TARGET_SR) for _word, first, _last in _word_spans(info["tokens"])]
    return sum(low <= at < high for at in starts for low, high in stretches)


async def _redo_stalled(
    request: Request, files: Sequence[_PreparedAudio], results: Sequence[Any], model_key: str
) -> List[Any]:
    """`results`, every file's pieces in order, with each piece that skipped
    speech (_stalled) decoded again as just its range, without context, and
    the redo kept if it hears words where the first decode heard none and
    loses none overall (_REDO_MIN_WORDS). A kept piece's window and piece in
    `files` change to match. Finding and judging them run on the audio pool:
    for hours of audio, that is a scan of every token."""
    results = list(results)
    if all(prepared.windows == prepared.ranges for prepared in files):
        return results  # nothing decoded context: short audio, or PARAKEET_CHUNK_CONTEXT_SEC=0
    loop, pool = asyncio.get_running_loop(), request.app.state.audio_pool

    def find() -> List[Tuple[int, _PreparedAudio, int, List[Tuple[int, int]]]]:
        """(in results, file, in file, what it skipped) for each piece to redo."""
        found = []
        cursor = 0
        for prepared in files:
            count = len(prepared.pieces)
            stalled = _stalled(prepared, results[cursor : cursor + count])
            found += [(cursor + index, prepared, index, skipped) for index, skipped in stalled.items()]
            cursor += count
        return found

    redo = await loop.run_in_executor(pool, find)
    if not redo:
        return results
    pieces = []
    for _at, prepared, index, _skipped in redo:
        (start, end), (window_start, _window_end) = prepared.ranges[index], prepared.windows[index]
        pieces.append(prepared.pieces[index][start - window_start : end - window_start])
    again = await _infer(request, pieces, model_key)

    def heard() -> List[bool]:
        """Whether each redo hears new words where its piece skipped, and as many in all."""
        tail = int(_WORD_TAIL_SEC * TARGET_SR)
        keep = []
        for (at, prepared, index, skipped), result in zip(redo, again):
            (start, end), window_start = prepared.ranges[index], prepared.windows[index][0]
            # A stretch ends at the first decode's next word, which the redo,
            # on another 80 ms frame grid, may time a frame earlier.
            new = _words_in(result, start, [(low, high - tail) for low, high in skipped])
            before = _words_in(results[at], window_start, [(start, end)])
            keep.append(new >= _REDO_MIN_WORDS and _words_in(result, start, [(start, end)]) >= before)
        return keep

    kept = 0
    for (at, prepared, index, _skipped), piece, result, keep in zip(
        redo, pieces, again, await loop.run_in_executor(pool, heard)
    ):
        if keep:
            results[at] = result
            prepared.pieces[index], prepared.windows[index] = piece, prepared.ranges[index]
            kept += 1
    logger.warning(
        "%d of %d chunks skipped speech; decoded again without context, %d heard words there and were kept",
        len(redo),
        len(results),
        kept,
    )
    return results


def _verbose_json(
    language: str,
    duration: float,
    text: str,
    segments: Sequence[Dict[str, Any]],
    words: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """The `verbose_json` response, in OpenAI's shape."""
    return {
        "task": "transcribe",
        "language": language,
        "duration": duration,
        "text": text,
        "segments": [
            {
                "id": index,
                "seek": 0,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["segment"],
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": 0.0,
            }
            for index, segment in enumerate(segments)
        ],
        "words": words,
    }


def _ok(model: Any, description: str, example: Any = None) -> Dict[str, Any]:
    """A response for the docs: its schema, and an example to show instead of
    one Swagger UI makes up from the schema ("string", "additionalProp1")."""
    response: Dict[str, Any] = {"model": model, "description": description}
    if example is not None:
        response["content"] = {"application/json": {"example": example}}
    return response


def _error(description: str, example: str) -> Dict[str, Any]:
    """An error response for the docs: `{"detail": example}`."""
    return _ok(ErrorResponse, description, {"detail": example})


def _form_doc(**extra: Any) -> Callable[[Dict[str, Any]], None]:
    """json_schema_extra for an optional form field. A form leaves a field out
    rather than sending null, so drop the null that Swagger UI shows as
    `string | (string | null)`; then add `extra` (an enum, say)."""

    def update(schema: Dict[str, Any]) -> None:
        branches = [branch for branch in schema.get("anyOf", []) if branch.get("type") != "null"]
        if len(branches) == 1:
            del schema["anyOf"]
            schema.update(branches[0])
        schema.update(extra)

    return update


_MODEL_DOC = (
    "The model, e.g. `parakeet-v3`, with a precision after a colon if you like "
    "(`parakeet-v3:fp16`). `GET /v1/models` lists them."
)
_QUANTIZATION_DOC = (
    "The model's precision, instead of a suffix on `model`; if both are sent, they "
    "must agree. `fp32` if neither names one."
)
_SPOKEN_NUMBERS_DOC = (
    "Write numbers, money and units in words, the way they were said (English "
    "only). Defaults to the server's `PARAKEET_SPOKEN_NUMBERS`."
)
_MODEL_ERRORS = {
    503: _error(
        "The service is still starting, or the model could not be loaded "
        "(`detail` names it and why); the next request tries again.",
        "Model is not ready",
    ),
}

_EXAMPLE_TEXT = "The quick brown fox jumps over the lazy dog."
_EXAMPLE_SEGMENTS = [{"start": 0.0, "end": 2.3473125, "segment": _EXAMPLE_TEXT}]
_EXAMPLE_WORDS = [
    {"start": start, "end": end, "word": word}
    for start, end, word in [
        (0.0, 0.16, "The"),
        (0.16, 0.4, "quick"),
        (0.4, 0.72, "brown"),
        (0.72, 1.04, "fox"),
        (1.04, 1.36, "jumps"),
        (1.36, 1.6, "over"),
        (1.6, 1.76, "the"),
        (1.76, 2.0, "lazy"),
        (2.0, 2.3473125, "dog."),
    ]
]
_TRANSCRIPTION_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    200: {
        "model": Union[Transcription, VerboseTranscription],
        "description": "The transcript, in the `response_format` asked for.",
        "content": {
            "application/json": {
                "examples": {
                    "json": {"summary": "json", "value": {"text": _EXAMPLE_TEXT}},
                    "verbose_json": {
                        "summary": "verbose_json, with timestamp_granularities[]=word",
                        "value": _verbose_json("auto", 2.3473125, _EXAMPLE_TEXT, _EXAMPLE_SEGMENTS, _EXAMPLE_WORDS),
                    },
                }
            },
            "text/plain": {"schema": {"type": "string"}, "example": _EXAMPLE_TEXT},
            "application/x-subrip": {"schema": {"type": "string"}, "example": _segments_to_srt(_EXAMPLE_SEGMENTS)},
            "text/vtt": {"schema": {"type": "string"}, "example": _segments_to_vtt(_EXAMPLE_SEGMENTS)},
        },
    },
    400: _error(
        "An unknown model, quantization, response_format, timestamp_granularities, "
        "language or aligner; a language the aligner doesn't align; `aligner_quantization`, "
        "now a suffix on `aligner`; or no file, or an empty one.",
        "language must be an ISO 639-1 code, e.g. 'en'; got 'English'",
    ),
    413: _error(
        "Over a request limit: the file's size, its decoded length, or the chunks it makes.",
        f"File exceeds the {MAX_UPLOAD_BYTES} byte upload limit",
    ),
    415: _error("The audio could not be decoded.", "Audio could not be decoded"),
    **_MODEL_ERRORS,
}
_BATCH_RESPONSES: Dict[Union[int, str], Dict[str, Any]] = {
    200: _ok(
        BatchTranscription,
        "Each file's transcript.",
        {
            "results": [{"filename": "fox.wav", "text": _EXAMPLE_TEXT, "duration": 2.3473125}],
            "batch_size": 1,
        },
    ),
    400: _error(
        "No files, an empty file, an unknown model, quantization or aligner, or "
        "`aligner_quantization`, now a suffix on `aligner`.",
        "No files provided",
    ),
    413: _error(
        "Over a request limit: too many files, too many bytes in all, or a file "
        "too big, too long or making too many chunks.",
        f"Batch contains {MAX_BATCH_FILES + 1} files; limit is {MAX_BATCH_FILES}",
    ),
    415: _error("A file could not be decoded; `detail` names it.", "fox.wav: audio could not be decoded"),
    **_MODEL_ERRORS,
}


_MODEL_CREATED = 1785888000  # catalog introduction (2026-08-05), fixed for stable output


def _model_card(name: str) -> Dict[str, Any]:
    config = MODEL_CONFIGS[name]
    return {
        "id": name,
        "object": "model",
        "created": _MODEL_CREATED,
        # The model's author; each quantization is someone's ONNX export of it.
        "owned_by": "openai" if _family(name) == "whisper" else "nvidia",
        "language": config["languages"],
        "quantizations": list(config["quantizations"]),
        "task": "automatic-speech-recognition",
    }


# Docs examples from the catalog, so they name a model that exists (a catalog
# has at least one; it may have no aligners).
_EXAMPLE_MODEL = next(iter(MODEL_CONFIGS))
_EXAMPLE_MODEL_CARD = _model_card(_EXAMPLE_MODEL)


@router.get(
    "/v1/models",
    tags=["models"],
    summary="List models",
    responses={
        200: _ok(
            ModelList,
            "Every model a request may name.",
            {"object": "list", "data": [_EXAMPLE_MODEL_CARD]},
        )
    },
)
def list_models():
    return {"object": "list", "data": [_model_card(name) for name in MODEL_CONFIGS]}


@router.get(
    "/v1/models/{model_id:path}",
    tags=["models"],
    summary="Retrieve a model",
    responses={
        200: _ok(ModelCard, "The model.", _EXAMPLE_MODEL_CARD),
        404: _error("No such model.", "Model 'parakeet' not found"),
    },
)
def retrieve_model(model_id: str):
    try:
        name = _validate_model(model_id)
    except HTTPException as exc:
        raise HTTPException(
            status_code=404, detail=f"Model {model_id!r} not found"
        ) from exc
    return _model_card(name)


def _aligner_card(name: str) -> Dict[str, Any]:
    config = ALIGNER_CONFIGS[name]
    return {
        "id": name,
        "object": "aligner",
        "created": _MODEL_CREATED,
        "language": list(config["languages"]),
        "quantizations": list(config["quantizations"]),
        "default_quantization": config["default_quantization"],
        "task": "forced-alignment",
    }


_EXAMPLE_ALIGNER = next(iter(ALIGNER_CONFIGS), None)
_EXAMPLE_ALIGNER_CARD = _aligner_card(_EXAMPLE_ALIGNER) if _EXAMPLE_ALIGNER else None


@router.get(
    "/v1/aligners",
    tags=["models"],
    summary="List aligners",
    responses={
        200: _ok(
            AlignerList,
            "Every aligner a request may name.",
            {"object": "list", "data": [_EXAMPLE_ALIGNER_CARD]} if _EXAMPLE_ALIGNER_CARD else None,
        )
    },
)
def list_aligners():
    """The aligners a request may name (`aligner`), as /v1/models lists models."""
    return {"object": "list", "data": [_aligner_card(name) for name in ALIGNER_CONFIGS]}


@router.get(
    "/v1/aligners/{aligner_id:path}",
    tags=["models"],
    summary="Retrieve an aligner",
    responses={
        200: _ok(AlignerCard, "The aligner.", _EXAMPLE_ALIGNER_CARD),
        404: _error("No such aligner.", "Aligner 'wav2vec2' not found"),
    },
)
def retrieve_aligner(aligner_id: str):
    name = aligner_id.strip().lower()
    if name not in ALIGNER_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Aligner {aligner_id!r} not found")
    return _aligner_card(name)


_EXAMPLE_KEY = variant_key(_EXAMPLE_MODEL)
_EXAMPLE_HEALTH = {
    "status": "healthy",
    "ready": True,
    "models": list(MODEL_CONFIGS),
    "loaded": [_EXAMPLE_KEY],
    "runtime": {
        _EXAMPLE_KEY: {
            "backend": "cpu",
            "sessions": {
                "asr._encoder": ["CPUExecutionProvider"],
                "asr._decoder_joint": ["CPUExecutionProvider"],
            },
            "fallback_reason": None,
        }
    },
    "cpu": CPU_INFO,
    "aligner": aligner.status(),
}


@router.get(
    "/health",
    tags=["health"],
    summary="Service status",
    responses={200: _ok(Health, "Always 200, ready or not.", _EXAMPLE_HEALTH)},
)
def health(request: Request):
    """Whether the service is ready, which models are loaded and what they
    run on, the CPU counts it sized its thread pools from, and each aligner's
    state. For a readiness probe, use `/healthz`."""
    ready = bool(getattr(request.app.state, "ready", False))
    return {
        "status": "healthy" if ready else "starting",
        "ready": ready,
        "models": list(MODEL_CONFIGS.keys()),
        "loaded": loaded_models(),
        "runtime": runtime_status(),
        "cpu": CPU_INFO,
        "aligner": aligner.status(),
    }


@router.get("/compare", include_in_schema=False)
def compare_page():
    """The page for comparing models and aligners by ear (PARAKEET_COMPARE_UI):
    a client of the routes here, which runs each row through /v1/audio/transcriptions."""
    if not COMPARE_UI:
        raise HTTPException(status_code=404, detail="Not Found")
    return FileResponse(Path(__file__).with_name("compare.html"), media_type="text/html")


@router.get(
    "/healthz",
    tags=["health"],
    summary="Readiness probe",
    responses={200: _ok(Ready, "Ready."), 503: _error("Not ready yet.", "not ready")},
)
def healthz(request: Request):
    """200 once the models in `PARAKEET_PRELOAD_MODELS` are loaded and warmed
    up, 503 until then."""
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ok"}


def _validate_language(language: Optional[str]) -> None:
    """A request's `language` is a bare ISO 639-1 code, or empty / "auto"
    (the default); anything else is a 400, whether or not it would be used."""
    code = (language or "").strip()
    if code not in ("", "auto") and not LANGUAGE_CODE.fullmatch(code):
        raise HTTPException(
            status_code=400, detail=f"language must be an ISO 639-1 code, e.g. 'en'; got {language!r}"
        )


def _validate_aligner(value: Optional[str]) -> Optional[Tuple[str, str]]:
    """The request's `aligner`, "name" or "name:quantization", as (name,
    quantization), or None if it names none: in the catalog, at one of its
    quantizations (else its default_quantization)."""
    if value is None:
        return None
    name, quantization = _named(value, "aligner")
    normalized = name.strip().lower()
    if normalized not in ALIGNER_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown aligner {name!r}. Available aligners: {sorted(ALIGNER_CONFIGS)}",
        )
    spec = ALIGNER_CONFIGS[normalized]
    quant = (quantization or spec["default_quantization"]).strip().lower()
    if quant not in spec["quantizations"]:
        raise HTTPException(
            status_code=400,
            detail=f"Aligner {normalized!r} has no {quant!r} quantization. Available: {list(spec['quantizations'])}",
        )
    return normalized, quant


def _validate_aligner_language(name: str, language: Optional[str]) -> None:
    """For a word request: the aligner must align its language, if it has one
    (none, with PARAKEET_ALIGN_DEFAULT_LANGUAGE empty: nothing is aligned)."""
    code = aligner.language_code(language)
    if code and not aligner.aligns(name, language):
        raise HTTPException(
            status_code=400,
            detail=f"Aligner {name!r} does not align {code!r}; it aligns {list(ALIGNER_CONFIGS[name]['languages'])}",
        )


def _transcript_language(model_name: str, language: Optional[str]) -> Optional[str]:
    """The transcript's language, for aligning and saying numbers: an
    English-only model's words are English, whatever `language` says."""
    return "en" if MODEL_CONFIGS[model_name]["languages"] == ["en"] else language


def _retimes(retime_words: Optional[bool]) -> bool:
    """Move word times out of pauses: the request's `retime_words`, else the
    server's PARAKEET_RETIME_WORDS."""
    return RETIME_WORDS if retime_words is None else retime_words


def _speaks(spoken_numbers: Optional[bool], language: Optional[str]) -> bool:
    """Say numbers in words: the request's `spoken_numbers`, else the server's
    PARAKEET_SPOKEN_NUMBERS; English only."""
    wanted = SPOKEN_NUMBERS if spoken_numbers is None else spoken_numbers
    return wanted and aligner.language_code(language) == "en"


async def _plain_granularities(request: Request) -> Optional[List[str]]:
    """`timestamp_granularities` without the brackets OpenAI's SDKs add. Read
    from the form rather than declared with Form(), so the docs show the
    field once."""
    return (await request.form()).getlist("timestamp_granularities") or None


async def _no_aligner_quantization(request: Request) -> None:
    """A 400 for `aligner_quantization`, gone before 2.0.0 but in the `latest`
    images for six days: FastAPI ignores a field it doesn't know, so its
    precision would quietly become the aligner's default. It says what to
    send instead."""
    form = await request.form()
    if "aligner_quantization" in form:
        name = str(form.get("aligner") or "<name>").partition(":")[0].strip()
        raise HTTPException(
            status_code=400,
            detail=f"aligner_quantization is gone: send aligner={name}:{str(form['aligner_quantization']).strip()} instead",
        )


@router.post(
    "/v1/audio/transcriptions",
    tags=["transcription"],
    summary="Transcribe audio",
    responses=_TRANSCRIPTION_RESPONSES,
    dependencies=[Depends(_no_aligner_quantization)],
)
async def transcribe(
    request: Request,
    file: UploadFile = File(..., description="The audio: anything FFmpeg can decode."),
    model: str = Form(..., description=_MODEL_DOC, examples=[_EXAMPLE_MODEL]),
    quantization: Optional[str] = Form(None, description=_QUANTIZATION_DOC, json_schema_extra=_form_doc()),
    response_format: str = Form(
        "json",
        description="`json` is just the text; `verbose_json` adds segments, and words if asked for.",
        json_schema_extra={"enum": list(_ALLOWED_FORMATS)},
    ),
    timestamp_granularities: Optional[List[str]] = Form(
        None,
        alias="timestamp_granularities[]",
        description=(
            "`word` adds `words` to a `verbose_json` response. `segment` changes nothing: "
            "`verbose_json` always has segments. Repeat the field to send both; the name "
            "without `[]` works too."
        ),
        json_schema_extra=_form_doc(items={"type": "string", "enum": sorted(_GRANULARITIES)}),
    ),
    timestamp_granularities_plain: Optional[List[str]] = Depends(_plain_granularities),
    language: Optional[str] = Form(
        None,
        description=(
            "The audio's language as an ISO 639-1 code (`en`), or `auto` (the default). "
            "The model is not told it: it is for aligning words and `spoken_numbers`, "
            "and `verbose_json` echoes it."
        ),
        json_schema_extra=_form_doc(),
    ),
    prompt: Optional[str] = Form(
        None, description="Accepted for OpenAI clients, and ignored.", json_schema_extra=_form_doc()
    ),
    temperature: Optional[float] = Form(
        None, description="Accepted for OpenAI clients, and ignored.", json_schema_extra=_form_doc()
    ),
    spoken_numbers: Optional[bool] = Form(None, description=_SPOKEN_NUMBERS_DOC, json_schema_extra=_form_doc()),
    aligner_name: Optional[str] = Form(
        None,
        alias="aligner",
        description=(
            "Time the words from the audio with a forced aligner, e.g. `wav2vec2-base-960h`, "
            "with a precision after a colon if you like. `GET /v1/aligners` lists them. "
            "None by default: words keep the model's own times, and Whisper has no words."
        ),
        json_schema_extra=_form_doc(),
    ),
    retime_words: Optional[bool] = Form(
        None,
        description=(
            "Move word times out of pauses, found by loudness. Defaults to the server's "
            "`PARAKEET_RETIME_WORDS`."
        ),
        json_schema_extra=_form_doc(),
    ),
):
    """Transcribe one audio file. OpenAI's SDKs work against it: send
    `quantization`, `spoken_numbers`, `aligner` and `retime_words`, which are this
    server's own, in `extra_body`."""
    del prompt, temperature  # accepted for OpenAI client compatibility
    model, quantization = _named(model, "model", quantization)
    model_name = _validate_model(model)
    model_key = _variant(model_name, quantization)
    family = _family(model_name)
    target_sec, max_sec, min_sec, context_sec = _chunk_bounds(model_name)
    output_format = _validate_format(response_format)
    granularities = set(timestamp_granularities or []) | set(
        timestamp_granularities_plain or []
    )
    if unknown := granularities - _GRANULARITIES:
        raise HTTPException(
            status_code=400,
            detail=f"timestamp_granularities takes {sorted(_GRANULARITIES)}; got {sorted(unknown)}",
        )
    _validate_language(language)
    heard = _transcript_language(model_name, language)
    # Naming an aligner is asking for aligned word times; there is no default.
    choice = _validate_aligner(aligner_name)
    speak = _speaks(spoken_numbers, heard)
    # Word timestamps: Parakeet emits them from its TDT tokens. Whisper returns
    # text only, so its words come purely from forced-aligning the transcript,
    # and only when the request names an aligner and the language is known —
    # never auto-detected for a multilingual Whisper request.
    want_words = output_format == "verbose_json" and "word" in granularities and (
        family == "parakeet"
        or (family == "whisper" and choice is not None and (heard or "").strip() not in ("", "auto"))
    )
    if want_words and choice is not None:
        _validate_aligner_language(choice[0], heard)
    raw = await _read_upload_limited(file)

    started = time.perf_counter()
    prepared = await _prepare_in_pool(request, raw, target_sec, max_sec, min_sec, context_sec)
    decode_ms = (time.perf_counter() - started) * 1000

    infer_started = time.perf_counter()
    results = await _infer(request, prepared.pieces, model_key)
    results = await _redo_stalled(request, [prepared], results, model_key)
    infer_ms = (time.perf_counter() - infer_started) * 1000

    stitch_started = time.perf_counter()
    full_text, segments, words = await _stitch_request(
        request,
        prepared,
        results,
        align=want_words and choice is not None,
        speak=speak,
        language=heard,
        aligner_choice=choice,
        retime_words=want_words and _retimes(retime_words),
    )
    stitch_ms = (time.perf_counter() - stitch_started) * 1000

    logger.info(
        "transcribe model=%s dur=%.2fs chunks=%d decode=%.0fms infer=%.0fms "
        "stitch=%.0fms total=%.0fms",
        model_key,
        prepared.duration,
        len(prepared.pieces),
        decode_ms,
        infer_ms,
        stitch_ms,
        (time.perf_counter() - started) * 1000,
    )

    if output_format == "text":
        return PlainTextResponse(full_text)
    if output_format == "srt":
        return Response(_segments_to_srt(segments), media_type="application/x-subrip")
    if output_format == "vtt":
        return Response(_segments_to_vtt(segments), media_type="text/vtt")
    if output_format == "verbose_json":
        return JSONResponse(
            _verbose_json(
                (language or "").strip() or "auto",
                prepared.duration,
                full_text,
                segments,
                words if want_words else None,
            )
        )
    return JSONResponse({"text": full_text})


@router.post(
    "/v1/audio/transcriptions/batch",
    tags=["transcription"],
    summary="Transcribe several files",
    responses=_BATCH_RESPONSES,
    dependencies=[Depends(_no_aligner_quantization)],
)
async def transcribe_batch(
    request: Request,
    files: List[UploadFile] = File(..., description="The audio files: repeat the field for each."),
    model: str = Form(..., description=_MODEL_DOC, examples=[_EXAMPLE_MODEL]),
    quantization: Optional[str] = Form(None, description=_QUANTIZATION_DOC, json_schema_extra=_form_doc()),
    spoken_numbers: Optional[bool] = Form(None, description=_SPOKEN_NUMBERS_DOC, json_schema_extra=_form_doc()),
    aligner_name: Optional[str] = Form(
        None,
        alias="aligner",
        description="Only to hear numbers for `spoken_numbers`: a batch returns no word times.",
        json_schema_extra=_form_doc(),
    ),
):
    """Transcribe several files with one model in one request. Text only: no
    `language`, `response_format` or word times."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"Batch contains {len(files)} files; limit is {MAX_BATCH_FILES}",
        )
    model, quantization = _named(model, "model", quantization)
    model_name = _validate_model(model)
    model_key = _variant(model_name, quantization)
    choice = _validate_aligner(aligner_name)  # it only hears numbers here
    target_sec, max_sec, min_sec, context_sec = _chunk_bounds(model_name)
    filenames = [upload.filename or "unnamed" for upload in files]

    raws: List[bytes] = []
    total_bytes = 0
    for upload in files:
        raw = await _read_upload_limited(upload)
        total_bytes += len(raw)
        if total_bytes > MAX_BATCH_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Batch exceeds the {MAX_BATCH_BYTES} byte limit",
            )
        raws.append(raw)

    loop = asyncio.get_running_loop()
    futures = [
        loop.run_in_executor(
            request.app.state.audio_pool,
            _prepare_audio,
            raw,
            target_sec,
            max_sec,
            min_sec,
            context_sec,
        )
        for raw in raws
    ]
    prepared_or_errors = await asyncio.gather(*futures, return_exceptions=True)
    prepared_files: List[_PreparedAudio] = []
    for filename, item in zip(filenames, prepared_or_errors):
        if isinstance(item, _AudioTooLong):
            raise HTTPException(status_code=413, detail=f"{filename}: {item}")
        if isinstance(item, BaseException):
            logger.exception(
                "batch audio preprocessing failed for %s",
                filename,
                exc_info=(type(item), item, item.__traceback__),
            )
            raise HTTPException(
                status_code=415, detail=f"{filename}: audio could not be decoded"
            )
        prepared_files.append(item)

    total_chunks = sum(len(item.pieces) for item in prepared_files)
    if total_chunks > MAX_REQUEST_CHUNKS:
        raise HTTPException(
            status_code=413,
            detail=f"Batch produced {total_chunks} chunks; limit is {MAX_REQUEST_CHUNKS}",
        )

    flattened = [piece for item in prepared_files for piece in item.pieces]
    flat_results = await _infer(request, flattened, model_key)
    flat_results = await _redo_stalled(request, prepared_files, flat_results, model_key)

    # The batch endpoint takes no `language`: the default decides, unless the
    # model is English-only.
    heard = _transcript_language(model_name, None)
    speak = _speaks(spoken_numbers, heard)
    cursor = 0
    response_items = []
    for filename, prepared in zip(filenames, prepared_files):
        count = len(prepared.pieces)
        item_results = flat_results[cursor : cursor + count]
        cursor += count
        text, _segments, _words = await _stitch_request(
            request, prepared, item_results, speak=speak, language=heard, aligner_choice=choice
        )
        response_items.append(
            {"filename": filename, "text": text, "duration": prepared.duration}
        )

    if cursor != len(flat_results):
        raise RuntimeError("inference result accounting mismatch")
    return {"results": response_items, "batch_size": len(response_items)}
