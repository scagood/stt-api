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


def _turn(seconds, level_db):
    """A speaker's turn: 250 ms syllables of a 220 Hz tone at level_db (dBFS
    RMS), with 100 ms dips to -70 dBFS between them."""
    rng = np.random.default_rng(0)
    syllable = np.sin(2 * np.pi * 220 * np.arange(SR // 4) / SR) * np.sqrt(2) * 10 ** (level_db / 20)
    dip = rng.standard_normal(SR // 10) * 10 ** (-70 / 20)
    return np.concatenate([part for _ in range(round(seconds / 0.35)) for part in (syllable, dip)]).astype(np.float32)


def test_pauses_are_quiet_stretches_of_200_ms_or_more():
    wav = _speech_and_pauses((1.0, True), (0.5, False), (1.0, True), (0.1, False), (1.0, True))
    assert retime.pauses(wav) == [(1.0, 1.5)]  # the 0.1 s dip is no pause


def test_a_quieter_speaker_is_no_pause():
    # 50 s at -20 dBFS, 10 s of a speaker 10 dB quieter, 50 s at -20 dBFS.
    # Under 0.6x the file's average throughout, the quiet turn was one 10 s
    # pause, and every word in it went to its edge.
    wav = np.concatenate([_turn(50, -20), _turn(10, -30), _turn(50, -20)])
    assert retime.pauses(wav) == []


def test_a_long_pause_of_room_tone_is_still_one():
    # Room tone over -60 dBFS never rises 10 dB over its own floor.
    tone = np.random.default_rng(1).standard_normal(5 * SR) * 10 ** (-50 / 20)
    wav = np.concatenate([_turn(20, -20), tone.astype(np.float32), _turn(20, -20)])
    assert [(round(a, 1), round(b, 1)) for a, b in retime.pauses(wav)] == [(19.9, 24.9)]


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


def test_a_pause_holding_several_words_leaves_them_alone():
    # Parakeet heard six words in this "pause": it is quiet speech. Moving
    # each to the pause's end piled them into 40 ms slivers after "three"
    # (18.0-18.8), pushing "nine" to 19.0-19.04.
    words = [_w(word, 12.0 + i, 12.8 + i) for i, word in enumerate(["three", "four", "five", "six", "seven", "eight"])]
    words.append(_w("nine", 18.0, 18.8))
    assert _times(retime.retime(words, [(12.0, 18.0)], 0.0, 30.0)) == _times(words)


def test_words_that_would_cross_in_a_pause_are_left_alone():
    # "said" goes after the pause, then "him." before it: moved, they crossed
    # and piled up at its end (said 18.0-18.4, him. 18.4-18.44).
    words = [_w("Then", 10.0, 11.8), _w("said", 13.0, 13.4), _w("him.", 14.0, 14.4), _w("Next", 18.5, 19.0)]
    assert _times(retime.retime(words, [(12.0, 18.0)], 0.0, 30.0)) == _times(words)


def test_one_word_inside_a_pause_on_each_side_still_moves():
    # The word before a pause slips late into it, the one after it early.
    words = [_w("him.", 2.2, 2.5), _w("The", 2.6, 2.8)]
    assert _times(retime.retime(words, GAP, 0.0, 10.0)) == [("him.", 1.7, 2.0), ("The", 3.0, 3.2)]


def test_a_pause_over_the_whole_chunk_moves_nothing():
    # No speech in the chunk to move a word next to: it collapsed to 0-0.
    assert _times(retime.retime([_w("Oh.", 4.0, 4.5)], [(0.0, 10.0)], 0.0, 10.0)) == [("Oh.", 4.0, 4.5)]


def test_the_first_word_is_at_least_40_ms_too():
    # Moved before a pause that starts 10 ms into the chunk, it had 10 ms.
    assert _times(retime.retime([_w("Yes.", 0.3, 0.7)], [(0.01, 2.0)], 0.0, 10.0)) == [("Yes.", 0.0, 0.04)]


def test_the_words_given_are_left_as_they_were():
    words = [_w("late", 2.2, 2.5)]
    retime.retime(words, GAP, 0.0, 10.0)
    assert words == [_w("late", 2.2, 2.5)]


def test_a_quiet_word_reaching_past_a_phrases_reach_is_no_pause():
    # A quiet 1 s phrase in a pause, then a word reaching past the 3 s around
    # it: no pause starts inside the word. Kept frame by frame, one did at 3 s.
    rng = np.random.default_rng(1)
    pause = rng.standard_normal(20 * SR) * 10 ** (-55 / 20)
    for at, seconds in [(5.0, 1.0), (8.95, 0.4)]:
        n = int(seconds * SR)
        pause[int(at * SR): int(at * SR) + n] += rng.standard_normal(n) * 10 ** (-40 / 20)
    first = _turn(40, -20)
    word = (first.size / SR + 8.95, first.size / SR + 9.35)
    wav = np.concatenate([first, pause.astype(np.float32), _turn(40, -20)])
    assert not [(a, b) for a, b in retime.pauses(wav) if a < word[1] and b > word[0]]
