"""Moving Parakeet's word times out of the pauses they slip into (retime.py)."""
from __future__ import annotations

import numpy as np
import pytest

from parakeet_service import retime
from parakeet_service.config import TARGET_SR

SR = TARGET_SR


def _speech_and_pauses(*parts):
    """(seconds, loud) parts: a 220 Hz tone at -20 dBFS, or near silence."""
    rng = np.random.default_rng(0)
    out = []
    for seconds, loud in parts:
        n = int(seconds * SR)
        if loud:
            out.append(0.1 * np.sin(2 * np.pi * 220 * np.arange(n) / SR))
        else:
            out.append(1e-4 * rng.standard_normal(n))
    return np.concatenate(out).astype(np.float32)


def _w(word, start, end):
    return {"word": word, "start": start, "end": end}


def _times(words):
    return [(w["word"], round(w["start"], 3), round(w["end"], 3)) for w in words]


def test_pauses_are_quiet_stretches_of_200_ms_or_more():
    wav = _speech_and_pauses((1.0, True), (0.5, False), (1.0, True), (0.1, False), (1.0, True))
    assert retime.pauses(wav) == [(1.0, 1.5)]  # the 0.1 s dip is no pause


def test_within_clips_to_a_chunk():
    spans = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    ends = [end for _start, end in spans]
    assert retime.within(spans, ends, 1.5, 5.5) == [(1.5, 2.0), (3.0, 4.0), (5.0, 5.5)]
    assert retime.within(spans, ends, 2.0, 3.0) == []


GAP = [(2.0, 3.0)]  # a pause from 2 s to 3 s, inside a 0-10 s chunk


@pytest.mark.parametrize(
    ("word", "moved"),
    [
        (_w("late", 2.2, 2.5), ("late", 3.0, 3.3)),          # inside, no punctuation: after it
        (_w("him.", 2.2, 2.5), ("him.", 1.7, 2.0)),          # inside, closes a sentence: before it
        (_w("The", 2.6, 3.4), ("The", 3.0, 3.4)),            # starts in it: starts at its end
        (_w("gospel", 1.6, 2.4), ("gospel", 1.6, 2.0)),      # ends in it: ends at its start
        (_w("gospel.", 1.6, 3.5), ("gospel.", 1.6, 2.0)),    # spans it: ends at its start
        (_w("far", 4.0, 4.3), ("far", 4.0, 4.3)),            # nowhere near: untouched
    ],
)
def test_one_word_and_one_pause(word, moved):
    assert _times(retime.retime([word], GAP, 0.0, 10.0)) == [moved]


def test_a_pause_at_a_chunk_edge_decides_the_side():
    # Chunks are cut mid-pause: speech past the pause is in the next chunk.
    assert _times(retime.retime([_w("so", 9.2, 9.4)], [(9.0, 10.0)], 0.0, 10.0)) == [("so", 8.8, 9.0)]
    assert _times(retime.retime([_w("Then.", 0.2, 0.4)], [(0.0, 0.5)], 0.0, 10.0)) == [("Then.", 0.5, 0.7)]


def test_words_stay_in_order_and_inside_the_chunk():
    words = [_w("The", 2.2, 2.4), _w("Adventures", 2.4, 3.2), _w("of", 3.2, 3.3)]
    out = retime.retime(words, GAP, 0.0, 10.0)
    # "The" moves after the pause; "Adventures", starting in it too, follows at
    # least 40 ms long, and "of" after it.
    assert _times(out) == [("The", 3.0, 3.2), ("Adventures", 3.2, 3.24), ("of", 3.24, 3.3)]
    assert all(a["end"] <= b["start"] for a, b in zip(out, out[1:]))
    edge = retime.retime([_w("last", 9.9, 9.98), _w("word", 9.98, 10.0)], [], 0.0, 10.0)
    assert all(0.0 <= w["start"] <= w["end"] <= 10.0 for w in edge)


def test_the_words_given_are_left_as_they_were():
    words = [_w("late", 2.2, 2.5)]
    retime.retime(words, GAP, 0.0, 10.0)
    assert words == [_w("late", 2.2, 2.5)]
