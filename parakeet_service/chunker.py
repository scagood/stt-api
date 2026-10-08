"""Pause-aware audio chunking with strict size invariants."""
from __future__ import annotations

import bisect
import logging
import threading
from typing import List, NamedTuple, Tuple

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


def relative_gate(rms: np.ndarray, ratio: float) -> float:
    """`ratio` times the average 20 ms frame level in `rms`, never under -60 dBFS."""
    return max(1e-3, float(rms.mean()) * ratio)


def runs(frames: np.ndarray) -> np.ndarray:
    """Each run of true `frames`, as a row of its start and end (exclusive) index."""
    return np.flatnonzero(np.diff(np.concatenate(([0], frames.astype(np.int8), [0])))).reshape(-1, 2)


def _volume_speech_segments(wav: np.ndarray) -> List[Range]:
    """Spans louder than the gate (PARAKEET_VAD_GATE_DB, else 0.4x the average
    20 ms frame level), joined across dips shorter than VAD_MIN_SILENCE_MS and
    padded by VAD_SPEECH_PAD_MS as Silero's are."""
    if wav.size < FRAME:
        return [(0, wav.size)] if np.any(np.abs(wav) > 1e-4) else []

    rms = frame_rms(wav)
    if VAD_GATE_DB is None:
        threshold = relative_gate(rms, 0.4)
    else:
        threshold = 10.0 ** (VAD_GATE_DB / 20.0)
    loud = runs(rms > threshold)
    if not loud.size:
        return []
    minimum_silence_frames = max(1, int(VAD_MIN_SILENCE_MS / 20))
    # Each run opens a segment unless the dip before it is too short to be a pause.
    opens = np.concatenate(([True], loud[1:, 0] - loud[:-1, 1] >= minimum_silence_frames))
    closes = np.concatenate((opens[1:], [True]))
    pad = int(VAD_SPEECH_PAD_MS * TARGET_SR / 1000)
    return [
        (max(0, start * FRAME - pad), min(wav.size, end * FRAME + pad))
        for start, end in zip(loud[opens, 0].tolist(), loud[closes, 1].tolist())
    ]


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


def _split_oversized(start: int, end: int, maximum: int) -> List[Range]:
    """Split one range into the fewest parts <= maximum, all about the same
    length: where it is cut at all, each part is at least half of maximum.
    Cutting target-length parts left whatever remained, down to a sliver
    once the target is cut to maximum for context (parakeet-v2): a decode of
    mostly context, and Parakeet invents words in very short input."""
    if end <= start:
        return []
    count = -(-(end - start) // maximum)
    cuts = [start + (end - start) * index // count for index in range(count + 1)]
    return list(zip(cuts, cuts[1:]))


class Plan(NamedTuple):
    ranges: List[Range]  # where each piece's words come from
    windows: List[Range]  # the audio each piece decodes: its range and context
    speech: List[Range]  # what VAD heard as speech; nothing for a clip it skipped


def plan_chunks(
    wav: np.ndarray,
    *,
    target_sec: float,
    max_sec: float,
    min_sec: float = CHUNK_MIN_SEC,
    context_sec: float = 0.0,
) -> Plan:
    """Ordered, non-empty, bounded ranges in the original waveform, and the
    window of audio to decode for each (_windows).

    Short clips bypass VAD. Long clips with no detected speech return no ranges,
    allowing the API to skip expensive ASR inference for silence.

    Bounds are the model's own (models.yaml). A long clip's ranges leave room
    for `context_sec` more on either side within max_sec.
    """
    total = int(wav.size)
    if total <= 0:
        return Plan([], [], [])

    target = max(1, int(target_sec * TARGET_SR))
    maximum = max(target, int(max_sec * TARGET_SR))
    if total <= maximum:
        return Plan([(0, total)], [(0, total)], [])
    context = int(context_sec * TARGET_SR)
    own_maximum = max(1, maximum - 2 * context)
    target = min(target, own_maximum)
    minimum = min(target, max(0, int(min_sec * TARGET_SR)))

    segments = _normalize_segments(_speech_segments(wav), total)
    if not segments:
        return Plan([], [], [])

    trim_gap = max(1, int(CHUNK_TRIM_SILENCE_SEC * TARGET_SR))
    packed: List[Range] = []
    # VAD can miss a quiet first or last syllable (the energy fallback in
    # particular), so the first and last chunks reach up to trim_gap past the
    # detected speech: room for the syllable, without feeding the model the
    # long silences cut out below. Less where the speech fits own_maximum but
    # the margin too would not: _split_oversized would cut it in two, inside
    # the speech.
    current_start, current_end = max(0, segments[0][0] - trim_gap), segments[0][1]
    if current_end - segments[0][0] <= own_maximum:
        current_start = max(current_start, current_end - own_maximum)
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

        # Mid-way through the pause, or later in it where the next phrase then
        # fits own_maximum, but never past own_maximum from where the range's
        # last piece starts: _split_oversized would cut it into one more piece,
        # inside speech.
        last_piece = _split_oversized(current_start, current_end, own_maximum)[-1][0]
        cut = min(max((current_end + start) // 2, min(end - own_maximum, start)), last_piece + own_maximum)
        # Cut at this pause once the range is the minimum, or before then where
        # taking the next phrase too would pass own_maximum and the phrase fits
        # after the cut: _split_oversized would cut inside speech, which makes
        # Parakeet drop words (#69). With the target cut to own_maximum for
        # context (parakeet-v2), that is nearly every cut.
        if current_end - current_start >= minimum or end - current_start > own_maximum >= end - cut:
            if cut > current_start:
                packed.append((current_start, cut))
            current_start = cut
            current_end = end
        else:
            current_end = end

    last_end = min(total, current_end + trim_gap)
    if current_end - current_start <= own_maximum:  # as with the first chunk's margin
        last_end = min(last_end, current_start + own_maximum)
    if last_end > current_start:
        packed.append((current_start, last_end))

    output: List[Range] = []
    for start, end in packed:
        output.extend(_split_oversized(start, end, own_maximum))

    # Defensive invariant filter: malformed VAD output must never reach ORT.
    ranges = [
        (start, end)
        for start, end in output
        if 0 <= start < end <= total and end - start <= own_maximum
    ]
    return Plan(ranges, _windows(ranges, segments, context, maximum), segments)


def _windows(ranges: List[Range], speech: List[Range], context: int, maximum: int) -> List[Range]:
    """The audio to decode for each range: `context` samples more on each side
    where it meets the next range (a cut in a pause, or one by length in long
    speech), taken from that range, and at most `maximum` samples in all.
    Never past the neighbour's far end, nor into a long silence cut out
    between two ranges.

    A window that ends inside speech can make Parakeet stop early and drop
    the rest of it, tens of seconds of words; one that starts inside speech
    can make it skip a stretch in the middle (#69). So each edge that meets
    a neighbour goes in a pause beyond the cut, with speech between them:
    the nearest pause at least `context` from the cut, else the farthest that
    fits. Only when none fits is it `context` from the cut. The start may
    take half the room `maximum` leaves around the range, or all of it when
    the end meets no neighbour; the end has the rest.
    """
    if context <= 0:
        return list(ranges)
    # Each pause between two VAD segments: where the speech before it ends,
    # where the speech after it starts, and its middle, where a window edge
    # goes (as plan_chunks cuts).
    befores = [left[1] for left in speech[:-1]]
    afters = [right[0] for right in speech[1:]]
    middles = [(left[1] + right[0]) // 2 for left, right in zip(speech, speech[1:])]
    windows: List[Range] = []
    for index, (start, end) in enumerate(ranges):
        window_start, window_end = start, end
        meets_next = index + 1 < len(ranges) and ranges[index + 1][0] == end
        if index and ranges[index - 1][1] == start:
            previous_start = ranges[index - 1][0]
            room = maximum - (end - start)
            low = max(previous_start, start - (room // 2 if meets_next else room))
            pauses = []  # nearest the cut first
            pause = bisect.bisect_left(afters, start) - 1  # speech between it and the cut
            while pause >= 0 and middles[pause] >= low:
                pauses.append(middles[pause])
                pause -= 1
            joined = index > 1 and ranges[index - 2][1] == previous_start
            if not joined and previous_start >= low:
                pauses.append(previous_start)  # after a long silence, or the start of the audio
            window_start = next(
                (at for at in pauses if at <= start - context),
                pauses[-1] if pauses else max(low, start - context),
            )
        if meets_next:
            following_end = ranges[index + 1][1]
            high = min(following_end, window_start + maximum)
            pauses = []  # nearest the cut first
            pause = bisect.bisect_right(befores, end)  # speech between it and the cut
            while pause < len(middles) and middles[pause] <= high:
                pauses.append(middles[pause])
                pause += 1
            joined = index + 2 < len(ranges) and ranges[index + 2][0] == following_end
            if not joined and following_end <= high:
                pauses.append(following_end)  # before a long silence, or the end of the audio
            window_end = next(
                (at for at in pauses if at >= end + context),
                pauses[-1] if pauses else min(high, end + context),
            )
        windows.append((window_start, window_end))
    return windows


def slice_chunks(wav: np.ndarray, ranges: List[Range]) -> List[np.ndarray]:
    """Return zero-copy contiguous views for the selected ranges."""
    return [wav[start:end] for start, end in ranges if end > start]
