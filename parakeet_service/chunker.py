"""Pause-aware audio chunking with strict size invariants."""
from __future__ import annotations

import logging
import threading
from typing import List, Tuple

from .config import (
    CHUNK_MIN_SEC,
    CHUNK_TRIM_SILENCE_SEC,
    TARGET_SR,
    VAD,
    VAD_GATE_DB,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
    VAD_THRESHOLD,
    logger,
)

import numpy as np

Range = Tuple[int, int]
# Silero carries state from one window to the next, so threads sharing a model
# would have to take turns, and a short request's VAD would wait out a long
# file's whole pass (about 0.7 s per minute of audio). One model per thread.
_vad_local = threading.local()
_vad_logged: set = set()


def _log_once(level: int, message: str, *args) -> None:
    """Log `message` the first time any thread loads a VAD, not once per thread."""
    if message not in _vad_logged:
        _vad_logged.add(message)
        logger.log(level, message, *args)


def _load_vad():
    try:
        from silero_vad import load_silero_vad  # type: ignore

        model = load_silero_vad(onnx=True)
    except Exception as exc:
        _log_once(logging.WARNING, "Silero VAD unavailable (%s); falling back to energy VAD", exc)
        return "energy"
    _log_once(logging.INFO, "Loaded Silero VAD (ONNX backend)")
    return model


def _get_vad():
    """This thread's Silero model, or "energy" when Silero is unavailable."""
    model = getattr(_vad_local, "model", None)
    if model is None:
        model = _vad_local.model = _load_vad()
    return model


def _speech_segments(wav: np.ndarray) -> List[Range]:
    """Speech spans in `wav` as half-open sample ranges, by PARAKEET_VAD."""
    if VAD == "volume":
        return _volume_speech_segments(wav)
    return _silero_speech_segments(wav)


def _silero_speech_segments(wav: np.ndarray) -> List[Range]:
    model = _get_vad()
    if model == "energy":
        return _volume_speech_segments(wav)

    from silero_vad import get_speech_timestamps  # type: ignore
    import torch

    timestamps = get_speech_timestamps(
        torch.from_numpy(wav),
        model,
        sampling_rate=TARGET_SR,
        threshold=VAD_THRESHOLD,
        min_silence_duration_ms=VAD_MIN_SILENCE_MS,
        speech_pad_ms=VAD_SPEECH_PAD_MS,
        return_seconds=False,
    )
    return [(int(item["start"]), int(item["end"])) for item in timestamps]


FRAME = int(0.02 * TARGET_SR)  # loudness is measured in 20 ms frames
# Frames squared at a time: 1000 s of audio, so the scratch copy stays ~64 MB
# however long the file (a 17-hour book is ~3.9 GB of float32 samples).
_RMS_BLOCK = 50_000


def frame_rms(wav: np.ndarray) -> np.ndarray:
    """RMS of each whole 20 ms frame of `wav`."""
    count = wav.size // FRAME
    framed = wav[: count * FRAME].reshape(count, FRAME)
    blocks = [framed[i: i + _RMS_BLOCK] for i in range(0, count, _RMS_BLOCK)]
    if not blocks:
        return np.empty(0, dtype=wav.dtype)
    return np.concatenate([np.sqrt((b * b).mean(axis=1) + 1e-12) for b in blocks])


def _volume_speech_segments(wav: np.ndarray) -> List[Range]:
    """Spans louder than the gate (PARAKEET_VAD_GATE_DB, else 0.4x the average
    20 ms frame level), joined across dips shorter than VAD_MIN_SILENCE_MS and
    padded by VAD_SPEECH_PAD_MS as Silero's are."""
    frame = FRAME
    if wav.size < frame:
        return [(0, wav.size)] if np.any(np.abs(wav) > 1e-4) else []

    rms = frame_rms(wav)
    frame_count = rms.size
    if VAD_GATE_DB is None:
        threshold = max(1e-3, float(rms.mean()) * 0.4)
    else:
        threshold = 10.0 ** (VAD_GATE_DB / 20.0)
    voiced = rms > threshold
    minimum_silence_frames = max(1, int(VAD_MIN_SILENCE_MS / 20))

    segments: List[Range] = []
    index = 0
    while index < frame_count:
        if not voiced[index]:
            index += 1
            continue
        start = last = cursor = index
        silence = 0
        while cursor < frame_count:
            if voiced[cursor]:
                silence, last = 0, cursor
            else:
                silence += 1
                if silence >= minimum_silence_frames:
                    break
            cursor += 1
        segments.append((start * frame, min((last + 1) * frame, wav.size)))
        index = max(cursor, index + 1)
    pad = int(VAD_SPEECH_PAD_MS * TARGET_SR / 1000)
    return [(max(0, start - pad), min(wav.size, end + pad)) for start, end in segments]


def _normalize_segments(segments: List[Range], total: int) -> List[Range]:
    normalized: List[Range] = []
    for start, end in sorted(segments):
        start = min(total, max(0, int(start)))
        end = min(total, max(start, int(end)))
        if end <= start:
            continue
        if normalized and start <= normalized[-1][1]:
            previous_start, previous_end = normalized[-1]
            normalized[-1] = (previous_start, max(previous_end, end))
        else:
            normalized.append((start, end))
    return normalized


def _split_oversized(start: int, end: int, target: int, maximum: int) -> List[Range]:
    """Split one non-empty range while guaranteeing every part <= maximum."""
    if end <= start:
        return []
    parts: List[Range] = []
    cursor = start
    while end - cursor > maximum:
        cut = min(end, cursor + target)
        if cut <= cursor:
            cut = min(end, cursor + maximum)
        parts.append((cursor, cut))
        cursor = cut
    if end > cursor:
        parts.append((cursor, end))
    return parts


def auto_chunk(
    wav: np.ndarray,
    *,
    target_sec: float,
    max_sec: float,
    min_sec: float = CHUNK_MIN_SEC,
) -> List[Range]:
    """Return ordered, non-empty, bounded ranges in the original waveform.

    Short clips bypass VAD. Long clips with no detected speech return no ranges,
    allowing the API to skip expensive ASR inference for silence.

    Bounds are the model's own (models.yaml).
    """
    total = int(wav.size)
    if total <= 0:
        return []

    target = max(1, int(target_sec * TARGET_SR))
    maximum = max(target, int(max_sec * TARGET_SR))
    minimum = max(0, int(min_sec * TARGET_SR))
    if total <= maximum:
        return [(0, total)]

    segments = _normalize_segments(_speech_segments(wav), total)
    if not segments:
        return []

    trim_gap = max(1, int(CHUNK_TRIM_SILENCE_SEC * TARGET_SR))
    packed: List[Range] = []
    # VAD can miss a quiet first or last syllable (the energy fallback in
    # particular), so the first and last chunks reach up to trim_gap past the
    # detected speech: room for the syllable, without feeding the model the
    # long silences cut out below.
    current_start, current_end = max(0, segments[0][0] - trim_gap), segments[0][1]
    for start, end in segments[1:]:
        # Cut at long silences and skip them entirely: feeding multi-second
        # silence to the model degrades recognition of the following speech,
        # and VAD already pads each segment, so no speech is lost.
        if start - current_end >= trim_gap:
            packed.append((current_start, current_end))
            current_start, current_end = start, end
            continue
        if end - current_start <= target:
            current_end = end
            continue

        if current_end - current_start >= minimum:
            cut = min(total, max(current_end, (current_end + start) // 2))
            if cut > current_start:
                packed.append((current_start, cut))
            current_start = cut
            current_end = end
        else:
            current_end = end

    last_end = min(total, current_end + trim_gap)
    if last_end > current_start:
        packed.append((current_start, last_end))

    output: List[Range] = []
    for start, end in packed:
        output.extend(_split_oversized(start, end, target, maximum))

    # Defensive invariant filter: malformed VAD output must never reach ORT.
    return [
        (start, end)
        for start, end in output
        if 0 <= start < end <= total and end - start <= maximum
    ]


def slice_chunks(wav: np.ndarray, ranges: List[Range]) -> List[np.ndarray]:
    """Return zero-copy contiguous views for the selected ranges."""
    return [wav[start:end] for start, end in ranges if end > start]
