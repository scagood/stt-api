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


# A quiet stretch heard again at its own level (loud_frames) is speech where it
# is this far over its own quietest tenth: 10 dB, where a pause's room tone
# stays within a few dB of its floor.
_OVER_FLOOR = 10.0 ** (10.0 / 20.0)
_SYLLABLE = 5  # frames: 100 ms
# A sound shorter than this in a quiet stretch heard again is no speech, unless
# a longer one is near. Breaths and rustles in LibriVox narration's long pauses
# were 0.1-0.34 s, and decoded alone, made up a word ("yeah"); 26 dB down, a
# quieter reader's phrases were 0.58-1.18 s and the words between them
# 0.12-0.32 s. A quiet "Yes." alone in a long pause is lost with the breaths.
_HEARD_ENOUGH = 25  # frames: 0.5 s


def relative_gate(rms: np.ndarray, ratio: float) -> float:
    """`ratio` times the average 20 ms frame level in `rms`, never under -60 dBFS."""
    return max(1e-3, float(rms.mean()) * ratio)


def runs(frames: np.ndarray) -> np.ndarray:
    """Each run of true `frames`, as a row of its start and end (exclusive) index."""
    return np.flatnonzero(np.diff(np.concatenate(([0], frames.astype(np.int8), [0])))).reshape(-1, 2)


def _runs_of(frames: np.ndarray, shortest: int) -> List[Range]:
    """runs() at least `shortest` long, as (start, end) pairs."""
    return [(start, end) for start, end in runs(frames).tolist() if end - start >= shortest]


def _joined(spans: np.ndarray, dip: int) -> np.ndarray:
    """`spans` (rows of runs()) joined across dips shorter than `dip` frames."""
    if not spans.size:
        return spans
    # Each run opens a span unless the dip before it is too short to be a pause.
    opens = np.concatenate(([True], spans[1:, 0] - spans[:-1, 1] >= dip))
    closes = np.concatenate((opens[1:], [True]))
    return np.stack((spans[opens, 0], spans[closes, 1]), axis=1)


def _dip_frames() -> int:
    """The shortest pause, VAD_MIN_SILENCE_MS, in 20 ms frames."""
    return max(1, int(VAD_MIN_SILENCE_MS / 20))


def loud_frames(rms: np.ndarray, ratio: float, relisten: int) -> np.ndarray:
    """Which frames (their levels, `rms`) are louder than `ratio` x the level
    of the audio around them, and than -60 dBFS.

    That level is first the whole file's average (relative_gate), which a
    speaker much quieter than the rest (a remote guest, a phone leg) can sit
    under throughout, and be taken for one long pause. So each run of quiet
    frames at least `relisten` long is heard again at its own level: where
    100 ms of it is louder than `ratio` x its average, and 10 dB over its
    quietest tenth, it is loud too: in a sound (joined across dips shorter
    than VAD_MIN_SILENCE_MS) heard for half a second or more, or within
    `relisten` of one. Then again within each run still that long, until none
    changes. A pause holding only room tone, clicks, or breaths stays quiet,
    however long or many.
    """
    loud = rms > relative_gate(rms, ratio)
    todo = _runs_of(~loud, relisten)
    while todo:
        start, end = todo.pop()
        part = rms[start:end]
        gate = max(relative_gate(part, ratio), float(np.percentile(part, 10)) * _OVER_FLOOR)
        # The median of 100 ms, as a syllable lasts: a click or a knock, a
        # frame or two loud in a pause, never passes.
        padded = np.pad(part, _SYLLABLE // 2, mode="edge")
        heard = np.median(np.lib.stride_tricks.sliding_window_view(padded, _SYLLABLE), axis=1) > gate
        # Speech is a sound (joined across dips shorter than a pause) heard for
        # half a second or more, so that breaths spread through one pause never
        # add up to any; and the shorter sounds within `relisten` of one, a
        # quiet speaker's short words, which would otherwise leave a stretch
        # long enough to cut out between two of their phrases.
        near = np.zeros_like(heard)
        for a, b in _joined(runs(heard), _dip_frames()).tolist():
            if heard[a:b].sum() >= _HEARD_ENOUGH:
                near[max(0, a - relisten): b + relisten] = True
        kept = heard & near
        if kept.any():
            loud[start:end] = kept
            todo.extend((start + a, start + b) for a, b in _runs_of(~kept, relisten))
    return loud


def _volume_speech_segments(wav: np.ndarray) -> List[Range]:
    """Spans louder than the gate (PARAKEET_VAD_GATE_DB, else 0.4x the level
    around them, loud_frames), joined across dips shorter than
    VAD_MIN_SILENCE_MS and padded by VAD_SPEECH_PAD_MS as Silero's are.

    The file's own gate hears again any quiet stretch long enough for
    plan_chunks to cut out (CHUNK_TRIM_SILENCE_SEC), so a quieter speaker's
    turn is decoded rather than dropped. A fixed gate is the operator's: all
    under it is silence."""
    if wav.size < FRAME:
        return [(0, wav.size)] if np.any(np.abs(wav) > 1e-4) else []

    rms = frame_rms(wav)
    if VAD_GATE_DB is None:
        loud = runs(loud_frames(rms, 0.4, max(1, int(CHUNK_TRIM_SILENCE_SEC * TARGET_SR) // FRAME)))
    else:
        loud = runs(rms > 10.0 ** (VAD_GATE_DB / 20.0))
    pad = int(VAD_SPEECH_PAD_MS * TARGET_SR / 1000)
    return [
        (max(0, start * FRAME - pad), min(wav.size, end * FRAME + pad))
        for start, end in _joined(loud, _dip_frames()).tolist()
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


# A cut inside speech goes where the 80 ms around it is quietest, looked for
# every 5 ms: a gap between words, not a dip mid-vowel. At most this far from
# where an even split puts it, and an eighth of the range's maximum, so no
# part is much shorter than an even split's.
QUIET_CUT_SEARCH_SEC = 3.0
_QUIET_SPAN = int(0.08 * TARGET_SR)
_QUIET_STEP = int(0.005 * TARGET_SR)
# A point is quiet where its power is under this share of the range's median
# (6 dB down); all louder points cost the same, so a cut never wanders after
# a level that only flickers. What moving a cut a second costs, in the same
# share: as much as a point 20 dB under the median. Enough to keep cuts near
# the even split where none is quieter, never to keep one in a word.
_QUIET = 0.25
_QUIET_CUT_COST_PER_SEC = 0.01
# The silence before a long silence (or the audio's end) a range keeps past
# its speech when its edge gives way to let a cut reach a quieter point.
EDGE_KEEP_SEC = 1.0


def _split_oversized(
    start: int,
    end: int,
    maximum: int,
    wav: np.ndarray | None = None,
    latest_start: int | None = None,
    earliest_end: int | None = None,
) -> List[Range]:
    """Split one range into the fewest parts <= maximum, all about the same
    length: where it is cut at all, no part is under a quarter of maximum.
    Cutting target-length parts left whatever remained, down to a sliver
    once the target is cut to maximum for context (parakeet-v2): a decode of
    mostly context, and Parakeet invents words in very short input.

    With `wav`, each cut moves from the even split to the quietest point near
    it (_quiet_cuts): a gap between words, where a cut at an arbitrary point
    likely splits one. The range may then start as late as `latest_start`
    and end as early as `earliest_end` (silence it can give up) where that
    lets a cut reach a quieter point."""
    if end <= start:
        return []
    count = -(-(end - start) // maximum)
    cuts = [start + (end - start) * index // count for index in range(count + 1)]
    if wav is not None and count > 1:
        cuts = _quiet_cuts(wav, cuts, maximum, max(start, latest_start or start), min(end, earliest_end or end))
    return list(zip(cuts, cuts[1:]))


def _quiet_cuts(wav: np.ndarray, even: List[int], maximum: int, latest_start: int, earliest_end: int) -> List[int]:
    """`even` (an even split of wav[even[0]:even[-1]] into parts of half
    `maximum` or more) with each inner cut moved, within
    QUIET_CUT_SEARCH_SEC (or maximum / 8) of it, to where the power of the
    _QUIET_SPAN around it is least, plus _QUIET_CUT_COST_PER_SEC for each
    second moved; and the ends as late or early as `latest_start` and
    `earliest_end`, as far, where the cuts need it. Every part stays
    <= maximum, and none moves an eighth of maximum either end, so each
    stays over a quarter of it.

    The cuts are chosen together, the cheapest chain by dynamic programming,
    as each one's room depends on where its neighbours go. Scored by power,
    not decibels: a deep pause scores barely better than a gap between words,
    so the search seldom puts one cut in a word to put another in a pause.
    Where no point near a cut is _QUIET, it stays where the even split puts it."""
    start, end = even[0], even[-1]
    reach = min(int(QUIET_CUT_SEARCH_SEC * TARGET_SR), maximum // 8)
    rms = frame_rms(wav[start:end])
    scale = 1.0 / (float(np.median(rms)) ** 2 + 1e-12)

    def candidates(point: int, low: int, high: int) -> np.ndarray:
        """`point`, and every _QUIET_STEP from it, from `low` to `high`."""
        return np.concatenate((
            np.arange(point - (point - low) // _QUIET_STEP * _QUIET_STEP, point, _QUIET_STEP),
            np.arange(point, high + 1, _QUIET_STEP),
        ))

    def power(at: np.ndarray) -> np.ndarray:
        """Mean power of the _QUIET_SPAN around each of `at`, over the median's,
        or 1 where that is not _QUIET."""
        low = max(0, int(at[0]) - _QUIET_SPAN // 2)
        part = wav[low: min(wav.size, int(at[-1]) + _QUIET_SPAN // 2)].astype(np.float64)
        sums = np.concatenate(([0.0], np.cumsum(part * part)))
        first = np.clip(at - _QUIET_SPAN // 2 - low, 0, part.size)
        last = np.clip(at + _QUIET_SPAN // 2 - low, 0, part.size)
        level = (sums[last] - sums[first]) / np.maximum(last - first, 1) * scale
        return np.where(level < _QUIET, level, 1.0)

    stages = [candidates(start, start, min(latest_start, start + reach))]
    stages += [candidates(point, point - reach, point + reach) for point in even[1:-1]]
    stages.append(candidates(end, max(earliest_end, end - reach), end))
    costs = [_QUIET_CUT_COST_PER_SEC * np.abs(at - point) / TARGET_SR for point, at in zip(even, stages)]
    for index in range(1, len(stages) - 1):
        costs[index] = costs[index] + power(stages[index])

    # A cut's options are the previous cut's candidates no more than maximum
    # before it: a tail of them, sorted, so the cheapest is a suffix minimum.
    total, back = costs[0], []
    for previous, at, cost in zip(stages, stages[1:], costs[1:]):
        best = np.minimum.accumulate(total[::-1])[::-1]
        where = np.searchsorted(previous, at - maximum)
        valid = where < previous.size
        where = np.minimum(where, previous.size - 1)
        # index of the suffix minimum from each `where`
        order = np.arange(previous.size)
        arg = np.where(total == best, order, previous.size)
        arg = np.minimum.accumulate(arg[::-1])[::-1]
        back.append(arg[where])
        total = np.where(valid, cost + best[where], np.inf)
    pick = int(np.argmin(total))
    if not np.isfinite(total[pick]):
        return even
    chosen = [pick]
    for step in reversed(back):
        chosen.append(int(step[chosen[-1]]))
    chosen.reverse()
    return [int(at[index]) for at, index in zip(stages, chosen)]


def _split_all(wav: np.ndarray, packed: List[Range], speech: List[Range], maximum: int) -> List[Range]:
    """Each of the `packed` ranges, split where it passes `maximum`
    (_split_oversized) at the quietest points. A range so split may give up
    the silence at its ends: to the range beside it where they meet, as long
    as that needs no more pieces; else down to EDGE_KEEP_SEC past its speech."""
    keep = int(EDGE_KEEP_SEC * TARGET_SR)
    ends = [stop for _start, stop in speech]
    starts = [begin for begin, _stop in speech]
    packed = list(packed)
    output: List[Range] = []
    for index, (start, end) in enumerate(packed):
        if end - start <= maximum:
            output.append((start, end))
            continue
        first = max(start, speech[bisect.bisect_right(ends, start)][0])
        last = min(end, speech[bisect.bisect_left(starts, end) - 1][1])
        if output and output[-1][1] == start:
            latest_start = min(first, output[-1][0] + maximum)
        else:
            latest_start = max(start, first - keep)
            # FIX2: never give up a sound in the margin (VAD may have missed a quiet word)
            margin = frame_rms(wav[start:first])
            if margin.size:
                heard = np.flatnonzero(margin > float(np.percentile(margin, 10)) * _OVER_FLOOR)
                if heard.size:
                    latest_start = max(start, min(latest_start, start + int(heard[0]) * FRAME - int(VAD_SPEECH_PAD_MS * TARGET_SR / 1000)))
        following = packed[index + 1] if index + 1 < len(packed) else None
        if following and following[0] == end:
            length = following[1] - following[0]
            earliest_end = max(last, following[1] - _room(length, maximum))
        else:
            earliest_end = min(end, last + keep)
            margin = frame_rms(wav[last:end])
            if margin.size:
                heard = np.flatnonzero(margin > float(np.percentile(margin, 10)) * _OVER_FLOOR)
                if heard.size:
                    earliest_end = min(end, max(earliest_end, last + (int(heard[-1]) + 1) * FRAME + int(VAD_SPEECH_PAD_MS * TARGET_SR / 1000)))
        pieces = _split_oversized(start, end, maximum, wav, latest_start, earliest_end)
        if output and output[-1][1] == start:
            output[-1] = (output[-1][0], pieces[0][0])
        if following and following[0] == end:
            packed[index + 1] = (pieces[-1][1], following[1])
        output.extend(pieces)
    return output


def _room(speech: int, maximum: int) -> int:
    """The longest a range holding `speech` samples may be with silence
    either side: any longer, and _split_oversized cuts it into one more
    piece than the speech needs, that cut inside speech too (#69). For
    speech that fits `maximum`, that is `maximum`."""
    return -(-speech // maximum) * maximum


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
    # long silences cut out below. Less where the first range's speech, however
    # many phrases it takes in, has no _room for all of the margin before it.
    first, lead = segments[0][0], max(0, segments[0][0] - trim_gap)
    current_start, current_end = lead, segments[0][1]
    for start, end in segments[1:]:
        if not packed:  # the first range, as far as it reaches so far
            current_start = max(lead, current_end - _room(current_end - first, own_maximum))
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
        # fits own_maximum, but no later than the range has _room for.
        cut = min(
            max((current_end + start) // 2, min(end - own_maximum, start)),
            current_start + _room(current_end - current_start, own_maximum),
        )
        # Cut at this pause once the range is the minimum, or before then where
        # taking the next phrase too would pass own_maximum and the phrase fits
        # after the cut: _split_oversized would cut inside speech, which makes
        # Parakeet drop words (#69). With the target cut to own_maximum for
        # context (parakeet-v2), that is nearly every cut.
        # Also where the phrase does not fit, but the two sides need no more
        # pieces than the whole (whose first range is as long as its speech: its
        # lead margin gives way), so the pause takes a cut _split_oversized
        # would put in speech. Only while the side after the cut keeps trim_gap
        # of its _room spare, or what the whole would: the next cut, capped by
        # that _room, must still reach over a pause to the phrase after it, or
        # the range after opens on silence that can cost it a piece. And only
        # where the silence before the phrase costs that side no piece either,
        # or its even split could cut inside speech, and the side before is
        # 2 s or more: no sliver, at a short trim_gap.
        whole = end - (current_start if packed else first)
        after = end - cut
        if (
            current_end - current_start >= minimum
            or end - current_start > own_maximum >= after
            or (
                _room(cut - current_start, own_maximum) + _room(after, own_maximum) <= _room(whole, own_maximum)
                and _room(after, own_maximum) - after >= min(trim_gap, _room(whole, own_maximum) - whole)
                and _room(end - start, own_maximum) == _room(after, own_maximum)
                and cut - current_start >= 2 * TARGET_SR
            )
        ):
            if cut > current_start:
                packed.append((current_start, cut))
            current_start = cut
            current_end = end
        else:
            current_end = end

    if not packed:
        current_start = max(lead, current_end - _room(current_end - first, own_maximum))
    last_end = min(total, current_end + trim_gap, current_start + _room(current_end - current_start, own_maximum))
    if last_end > current_start:
        packed.append((current_start, last_end))

    output = _split_all(wav, packed, segments, own_maximum)

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
