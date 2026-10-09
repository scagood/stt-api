"""Parakeet's word times moved out of the pauses they slip into
(PARAKEET_RETIME_WORDS, `retime_words`).

Parakeet times words off its 80 ms decoder frames, and around a pause they
slip: the word before it starts after the speech has stopped, the word after
it before the speech starts again. Here pauses are found by loudness, and only
the words touching one move, to its edge; every other word keeps Parakeet's
time. A forced aligner times every word from the audio instead, better and at
several times the cost.
"""
from __future__ import annotations

import bisect
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .chunker import FRAME, frame_rms, loud_frames
from .config import TARGET_SR

Span = Tuple[float, float]

# A pause is at least this long and quieter than this share of the level
# around it (chunker.loud_frames: the file's average 20 ms frame level, never
# under -60 dBFS): against the aligner on three LibriVox chapters, these placed
# words best. The volume splitter's gate (0.4x) is lower: it looks for places
# to cut, not word edges.
_GATE_RATIO = 0.6
_MIN_PAUSE_SEC = 0.2
# A stretch quieter than that at least this long is heard again at its own
# level: a speaker much quieter than the file's average would otherwise be one
# long pause. From 2 s, breaths in narration's longer pauses were heard as
# speech and words moved to them: on three LibriVox chapters, 45 of the 58
# words that then moved went farther from the aligner's times.
_RELISTEN_SEC = 3.0
# No word is shortened to less than this, or left out of order.
_MIN_WORD_SEC = 0.04
# A word ending in one of these closes a sentence or clause: in a pause, it
# belongs before it.
_CLOSES = tuple(".,;:?!…\"'”’)")


def pauses(wav: np.ndarray) -> List[Span]:
    """The pauses in `wav`, as (start, end) seconds in order."""
    rms = frame_rms(wav)
    if not rms.size:
        return []
    quiet = (~loud_frames(rms, _GATE_RATIO, round(_RELISTEN_SEC * TARGET_SR / FRAME))).astype(np.int8)
    runs = np.flatnonzero(np.diff(np.concatenate(([0], quiet, [0])))).reshape(-1, 2)
    second = FRAME / TARGET_SR
    return [(a * second, b * second) for a, b in runs if (b - a) * second >= _MIN_PAUSE_SEC]


def within(spans: Sequence[Span], ends: Sequence[float], low: float, high: float) -> List[Span]:
    """The parts of the ordered `spans` (their `ends` listed alongside, for
    bisecting) that fall between low and high."""
    first = bisect.bisect_right(ends, low)
    out = []
    for start, end in spans[first:]:
        if start >= high:
            break
        out.append((max(start, low), min(end, high)))
    return out


def _before(word: Dict[str, Any], gap: Span, low: float, high: float) -> bool:
    """Whether a word wholly inside `gap` belongs before it rather than after."""
    if gap[1] >= high:
        return True
    if gap[0] <= low:
        return False
    return word["word"].endswith(_CLOSES)


def _crowded(gap: Span, words: Sequence[Dict[str, Any]], starts: Sequence[float], low: float, high: float) -> bool:
    """Whether `gap` holds speech rather than a pause: it is the whole chunk,
    with no speech beside it to move a word to, or the words wholly inside it
    are more than one going to a side, or one going after it then one going
    before. A pause has one word at each edge, the last before it and the
    first after it; anything else is speech the gate took for quiet, or times
    off by more than a word, and moving those words piles them up at an edge."""
    start, end = gap
    if start <= low and end >= high:
        return True
    before = after = 0
    for index in range(bisect.bisect_left(starts, start), len(words)):
        word = words[index]
        if word["start"] >= end:
            break
        if word["end"] <= end:
            side = _before(word, gap, low, high)
            if side and after:  # they would cross, and pile up
                return True
            before, after = before + side, after + (not side)
    return before > 1 or after > 1


def retime(words: List[Dict[str, Any]], gaps: Sequence[Span], low: float, high: float) -> List[Dict[str, Any]]:
    """One chunk's words (absolute seconds, in order), moved out of its `gaps`
    (from within(), the chunk being low..high). A word that starts in a pause
    starts at its end; one that ends in it, or spans it, ends at its start. A
    word wholly inside goes to the side it belongs to: before a pause that runs
    to the end of the chunk, after one that runs from its start (chunks are cut
    mid-pause, so the speech on the other side is in the next or last chunk),
    else before it if it closes a sentence or clause, after it if not. A gap
    that holds speech (_crowded) moves none of its words."""
    out = [dict(word) for word in words]
    starts = [word["start"] for word in out]
    gaps = [gap for gap in gaps if not _crowded(gap, out, starts, low, high)]
    k = 0
    for index, word in enumerate(out):
        while k < len(gaps) and gaps[k][1] <= word["start"]:
            k += 1
        for start, end in gaps[max(0, k - 1): k + 2]:
            first, last = word["start"], word["end"]
            if last <= start or first >= end:
                continue
            length = last - first
            if first >= start and last <= end:
                if _before(word, (start, end), low, high):
                    previous = out[index - 1]["end"] if index else low
                    word["start"], word["end"] = max(start - length, previous), start
                else:
                    word["start"], word["end"] = end, end + length
            elif first < start and last > end:
                word["end"] = start
            elif first >= start:
                word["start"], word["end"] = end, max(last, end)
            else:
                word["end"] = start
    if out:  # the first word too: moved before a pause, it can have none
        out[0]["end"] = max(out[0]["end"], out[0]["start"] + _MIN_WORD_SEC)
    for previous, word in zip(out, out[1:]):
        word["start"] = max(word["start"], previous["end"])
        word["end"] = max(word["end"], word["start"] + _MIN_WORD_SEC)
    for word in out:
        word["start"] = min(max(word["start"], low), high)
        word["end"] = min(max(word["end"], word["start"]), high)
    return out
