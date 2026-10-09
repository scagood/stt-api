from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from parakeet_service import chunker, routes

MAX_SEC = 75.0
BOUNDS = {"target_sec": 60.0, "max_sec": MAX_SEC}


def _ranges(wav, **bounds):
    """Where plan_chunks cuts `wav`."""
    return chunker.plan_chunks(wav, **bounds).ranges


def _assert_valid(ranges, total, maximum):
    previous_end = -1
    for start, end in ranges:
        assert 0 <= start < end <= total
        assert end - start <= maximum
        assert start >= previous_end
        previous_end = end


def test_empty_audio_has_no_chunks():
    assert _ranges(np.empty(0, dtype=np.float32), **BOUNDS) == []


def test_short_audio_bypasses_vad(monkeypatch):
    monkeypatch.setattr(
        chunker,
        "_speech_segments",
        lambda _wav: (_ for _ in ()).throw(AssertionError("VAD should not run")),
    )
    waveform = np.zeros(int(MAX_SEC * chunker.TARGET_SR) - 1)
    plan = chunker.plan_chunks(waveform, **BOUNDS)
    assert plan.ranges == [(0, waveform.size)]
    assert plan.speech is None  # not run, rather than none heard


def test_long_silence_skips_inference(monkeypatch):
    monkeypatch.setattr(chunker, "_speech_segments", lambda _wav: [])
    waveform = np.zeros(int((MAX_SEC + 10) * chunker.TARGET_SR))
    assert _ranges(waveform, **BOUNDS) == []


def test_long_uninterrupted_speech_has_no_phantom_tail(monkeypatch):
    total = int((MAX_SEC * 2.5) * chunker.TARGET_SR)
    monkeypatch.setattr(
        chunker, "_speech_segments", lambda _wav: [(0, total)]
    )
    ranges = _ranges(np.ones(total, dtype=np.float32), **BOUNDS)
    maximum = int(MAX_SEC * chunker.TARGET_SR)
    _assert_valid(ranges, total, maximum)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == total
    assert all(end - start > 1 for start, end in ranges)


def test_long_silence_gap_is_cut_out_of_chunks(monkeypatch):
    sr = chunker.TARGET_SR
    total = int(MAX_SEC * 2 * sr)
    speech = [(0, 10 * sr), (40 * sr, total)]  # 30 s silent gap
    monkeypatch.setattr(chunker, "_speech_segments", lambda _wav: speech)
    ranges = _ranges(np.ones(total, dtype=np.float32), **BOUNDS)
    maximum = int(MAX_SEC * sr)
    _assert_valid(ranges, total, maximum)
    assert ranges[0] == (0, 10 * sr)
    assert ranges[1][0] == 40 * sr
    assert not any(start < 40 * sr and end > 10 * sr for start, end in ranges[1:])


def test_bounds_override_caps_chunks(monkeypatch):
    # Whisper path: 90 s of continuous speech must chunk to <=30 s pieces, not
    # the 75 s Parakeet default (which would silently truncate under Whisper).
    sr = chunker.TARGET_SR
    total = int(90 * sr)
    monkeypatch.setattr(chunker, "_speech_segments", lambda _wav: [(0, total)])
    ranges = _ranges(
        np.ones(total, dtype=np.float32), target_sec=25.0, max_sec=30.0, min_sec=20.0
    )
    _assert_valid(ranges, total, int(30.0 * sr))
    assert ranges[0][0] == 0
    assert ranges[-1][1] == total


@pytest.mark.parametrize(
    ("seconds", "bounds", "own_maximum_sec"),
    [
        # parakeet-v2 with 5 s of context: target and maximum both 20 s; was 20 + 20 + 0.1 s
        (40.1, {"target_sec": 25.0, "max_sec": 30.0, "min_sec": 20.0, "context_sec": 5.0}, 20.0),
        (125.5, {**BOUNDS, "context_sec": 5.0}, 65.0),  # parakeet-v3: was 60 + 60 + 5.5 s
        (55.1, {"target_sec": 25.0, "max_sec": 30.0, "min_sec": 20.0}, 30.0),  # Whisper: was 25 + 25 + 5.1 s
    ],
)
def test_speech_with_no_pause_is_split_evenly(monkeypatch, seconds, bounds, own_maximum_sec):
    # One long stretch of speech (a fixed volume gate under the noise, say)
    # is cut by length. Target-length pieces left the rest as the last one,
    # a sliver once the target is cut to the maximum for context.
    total = int(seconds * SR)
    own_maximum = int(own_maximum_sec * SR)
    monkeypatch.setattr(chunker, "_speech_segments", lambda _wav: [(0, total)])
    ranges = _ranges(np.ones(total, dtype=np.float32), **bounds)
    _assert_valid(ranges, total, own_maximum)
    assert ranges[0][0] == 0 and ranges[-1][1] == total
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert len(ranges) == -(-total // own_maximum)  # no more pieces than fit
    lengths = [end - start for start, end in ranges]
    assert max(lengths) - min(lengths) <= 1
    assert min(lengths) > 5 * SR


@pytest.mark.parametrize(
    ("speech", "seconds", "expected"),
    [
        ([(0, 38.5)], 41.5, [(0, 20), (20, 40)]),  # the margin past the speech
        ([(0, 38.5)], 45, [(0, 20), (20, 40)]),
        ([(0, 39.8), (40.8, 45.8)], 50, [(0, 20), (20, 40), (40, 48.8)]),  # mid-way through a pause
        ([(3, 41.5)], 45, [(1.5, 21.5), (21.5, 41.5)]),  # the margin before the first phrase
        ([(3, 10), (10.5, 41.5)], 45, [(1.5, 21.5), (21.5, 41.5)]),  # before a first range of two
    ],
)
def test_silence_around_the_speech_adds_no_piece(monkeypatch, speech, seconds, expected):
    # parakeet-v2 with 5 s of context: 38.5 s of speech is two pieces of 20 s
    # at most. The silence after it, or before it, would make the range
    # 41.5 s, three equal pieces, both cuts inside speech: it spans 40 s instead.
    monkeypatch.setattr(chunker, "_speech_segments", lambda _wav: [_at(*span) for span in speech])
    ranges = chunker.plan_chunks(
        np.zeros(_at(seconds)[0], dtype=np.float32), target_sec=25.0, max_sec=30.0, min_sec=20.0, context_sec=5.0
    ).ranges
    assert ranges == [_at(*span) for span in expected]


def test_short_audio_is_one_piece_whatever_the_context():
    waveform = np.zeros(int(MAX_SEC * chunker.TARGET_SR))
    assert _ranges(waveform, **BOUNDS, context_sec=5.0) == [(0, waveform.size)]


def test_ranges_and_their_windows_fit_the_chunk(monkeypatch):
    sr = chunker.TARGET_SR
    total = int(MAX_SEC * 4 * sr)
    phrases = [(start, start + int(3.5 * sr)) for start in range(0, total, 4 * sr)]  # 0.5 s pauses
    for speech in (phrases, [(0, total)]):  # cut in pauses, or by length
        monkeypatch.setattr(chunker, "_speech_segments", lambda _wav, speech=speech: speech)
        ranges, windows, heard = chunker.plan_chunks(np.ones(total, dtype=np.float32), **BOUNDS, context_sec=5.0)
        assert heard == speech
        _assert_valid(ranges, total, int((MAX_SEC - 10) * sr))
        assert max(end - start for start, end in ranges) >= 59 * sr  # the 60 s target still fits
        assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
        assert windows[0][0] == 0 and windows[-1][1] == total
        for (start, end), (window_start, window_end) in zip(ranges, windows):
            assert window_start <= start < end <= window_end
            assert window_end - window_start <= MAX_SEC * sr
        for (_start, cut), (_window_start, window_end) in zip(ranges[:-1], windows[:-1]):
            if speech is phrases:  # mid-way through the first pause 5 s or more past the cut
                assert window_end % (4 * sr) == int(3.75 * sr) and 5 * sr <= window_end - cut < 9 * sr
            else:  # no pause to end in: 5 s past the cut
                assert window_end == cut + 5 * sr
        for (cut, _end), (window_start, _window_end) in zip(ranges[1:], windows[1:]):
            if speech is phrases:  # mid-way through a pause before the cut, as far back as fits
                assert window_start % (4 * sr) == int(3.75 * sr) and cut - window_start >= 4 * sr
            else:  # no pause to start in: 5 s before the cut
                assert window_start == cut - 5 * sr


def _ordinary_speech(seconds):
    """Phrases of 2-6 s with pauses of 0.5-1.5 s, and the total samples."""
    rng = np.random.default_rng(0)
    speech, at = [], 1.0
    while True:
        length = rng.uniform(2, 6)
        if at + length > seconds - 1:
            return speech, int(seconds * SR)
        speech.append(_at(at, at + length))
        at += length + rng.uniform(0.5, 1.5)


@pytest.mark.parametrize(
    ("name", "seconds", "most"),
    [
        # 51 chunks before, with 20 cuts inside speech
        ("parakeet-v2", 600, 34),
        # 623 before: over the 512 chunks a request may make, so a 413
        ("parakeet-v2", 2 * 3600, 408),
        ("parakeet-v3", 600, 11),
    ],
)
def test_ordinary_speech_is_cut_in_pauses_after_room_for_context(monkeypatch, name, seconds, most):
    # parakeet-v2's 25 s target is cut to the 20 s its 30 s maximum leaves
    # beside 5 s of context either side, so its 20 s minimum never came
    # first: ranges ran past 20 s, and were cut by length inside speech,
    # leaving slivers.
    monkeypatch.setattr(routes, "CHUNK_MIN_SEC", 20.0)
    monkeypatch.setattr(routes, "CHUNK_CONTEXT_SEC", 5.0)
    target_sec, max_sec, min_sec, context_sec = routes._chunk_bounds(name)
    own_maximum = int((max_sec - 2 * context_sec) * SR)
    own_target = min(int(target_sec * SR), own_maximum)
    speech, total = _ordinary_speech(seconds)
    monkeypatch.setattr(chunker, "_speech_segments", lambda _wav: speech)
    ranges = chunker.plan_chunks(
        np.broadcast_to(np.float32(0), (total,)),  # hours of samples in no memory
        target_sec=target_sec, max_sec=max_sec, min_sec=min_sec, context_sec=context_sec,
    ).ranges
    assert len(ranges) <= most
    _assert_valid(ranges, total, own_maximum)
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    cuts = [end for _start, end in ranges[:-1]]
    assert not any(start < cut < end for cut in cuts for start, end in speech)  # all in pauses
    # every range holds speech: no slivers of silence
    assert all(any(begin < end and start < stop for begin, stop in speech) for start, end in ranges)
    # each closes at the last pause before its target: short of it by less than a phrase and a pause
    assert all(end - start > own_target - 7.5 * SR for start, end in ranges[:-1])


@pytest.mark.parametrize(
    ("speech", "seconds", "expected"),
    [
        # 19.8 s at the pause, short of the minimum, but the next phrase would
        # pass 20 s: it closes in the pause, at 20 s rather than mid-way at
        # 20.2 s, and the last range's margin past the speech stops at 20 s
        ([(3, 19.8), (20.6, 39.5)], 45, [(0, 20), (20, 40)]),
        # the next phrase fits from late in the pause, not from mid-way
        ([(3, 13), (14, 33.8)], 40, [(0, 13.8), (13.8, 33.8)]),
        # the margin before the first phrase gives way to it
        ([(2, 21.5), (22.5, 30)], 40, [(1.5, 21.5), (21.5, 33)]),
        ([(2, 21.5), (26, 40)], 40, [(1.5, 21.5), (26, 40)]),  # before a long silence
        ([(2, 21.5)], 40, [(1.5, 21.5)]),  # the last range too
        # a phrase too long for a range is cut by length, but the pause after
        # it goes with the next range, not into a sliver
        ([(3, 40), (41, 50)], 60, [(0, 20), (20, 40), (40, 53)]),
        # 18.8 s at the pause, short of the minimum, and the next phrase does
        # not fit after it: still cut there, as both sides need no more pieces
        # than the whole, which would be cut evenly inside speech twice
        ([(0, 18.8), (21.5, 43.5)], 43.5, [(0, 20), (20, 31.75), (31.75, 43.5)]),
        # but not where the side after the cut has no room spare: the next cut
        # could not reach the last phrase, and the silence before it would
        # cost a piece
        ([(0, 10), (11.5, 51.45), (54, 93.9)], 97, [(0, 18), (18, 36), (36, 54), (54, 74), (74, 94)]),
    ],
)
def test_ranges_close_in_a_pause_rather_than_pass_their_maximum(monkeypatch, speech, seconds, expected):
    # parakeet-v2 with 5 s of context: ranges of 20 s at most, target 20 s,
    # minimum 20 s. None is cut inside a phrase that fits one, or leaves a
    # sliver of silence.
    monkeypatch.setattr(chunker, "_speech_segments", lambda _wav: [_at(*span) for span in speech])
    ranges = chunker.plan_chunks(
        np.zeros(_at(seconds)[0], dtype=np.float32), target_sec=25.0, max_sec=30.0, min_sec=20.0, context_sec=5.0
    ).ranges
    assert ranges == [_at(*span) for span in expected]


@pytest.mark.parametrize(
    ("trim_sec", "context_sec", "speech", "seconds", "in_speech"),
    [
        # a 1 s trim gap would let the cut at the pause leave a 1.5 s range;
        # the long phrase is cut evenly, twice, as without the pause
        (1.0, 5.0, [(0, 1), (1.5, 40.012)], 40.012, 2),
        # a 6 s trim gap and 15 s ranges: the silence before the last phrase
        # but one would cost the range after the cut a piece, its even split
        # a cut inside speech
        (6.0, 7.5, [(0, 8), (11, 15.05), (20, 24.3), (30, 40)], 40, 0),
    ],
)
def test_a_cut_before_the_minimum_leaves_no_sliver_or_cut_in_speech(
    monkeypatch, trim_sec, context_sec, speech, seconds, in_speech
):
    monkeypatch.setattr(chunker, "CHUNK_TRIM_SILENCE_SEC", trim_sec)
    speech = [_at(*span) for span in speech]
    monkeypatch.setattr(chunker, "_speech_segments", lambda _wav: speech)
    ranges = chunker.plan_chunks(
        np.zeros(_at(seconds)[0], dtype=np.float32),
        target_sec=25.0, max_sec=30.0, min_sec=20.0, context_sec=context_sec,
    ).ranges
    assert all(end - start >= 2 * SR for start, end in ranges)
    cuts = [cut for _start, cut in ranges[:-1]]
    assert sum(any(start < cut < end for start, end in speech) for cut in cuts) == in_speech


def _at(*seconds):
    return tuple(int(second * SR) for second in seconds)


@pytest.mark.parametrize(
    ("speech", "maximum", "first"),
    [
        # pauses mid-way at 12.25 s and 14.3 s: the first 3 s or more past the cut at 10 s
        ([(0, 9.5), (10.5, 12), (12.5, 14), (14.6, 20)], 15, (0, 14.3)),
        # only the 12.25 s pause fits in 13.5 s: less context, still ending in a pause
        ([(0, 9.5), (10.5, 12), (12.5, 14), (14.6, 20)], 13.5, (0, 12.25)),
        # no pause past the cut: 3 s past it, inside the speech
        ([(0, 9.5), (10.5, 20)], 15, (0, 13)),
    ],
)
def test_windows_end_in_a_pause_past_the_cut(speech, maximum, first):
    ranges = [_at(0, 10), _at(10, 20)]
    windows = chunker._windows(ranges, [_at(*span) for span in speech], _at(3)[0], _at(maximum)[0])
    assert windows == [_at(*first), _at(7, 20)]


def test_windows_start_in_a_pause_before_the_cut():
    # pauses mid-way at 5.25 s and 7.7 s, before the cut at 10 s
    ranges = [_at(0, 10), _at(10, 20)]
    speech = [_at(0, 5), _at(5.5, 7.5), _at(7.9, 9.5), _at(10.5, 20)]
    # the nearest 3 s or more before the cut
    assert chunker._windows(ranges, speech, _at(3)[0], _at(30)[0])[1] == _at(5.25, 20)
    # only 7.7 s fits in 14 s: the farthest that does
    assert chunker._windows(ranges, speech, _at(3)[0], _at(14)[0])[1] == _at(7.7, 20)
    # no pause before the cut: 3 s before it, inside the speech
    assert chunker._windows(ranges, [_at(0, 9.5), _at(10.5, 20)], _at(3)[0], _at(14)[0])[1] == _at(7, 20)


def test_context_stays_out_of_long_silences():
    # cuts at 10 s and 40 s; 12-30 s is a long silence, cut out. The audio's
    # start and the long silence's end count as pauses to start in.
    ranges = [_at(0, 10), _at(10, 12), _at(30, 40), _at(40, 42)]
    speech = [_at(0, 9.5), _at(10.5, 12), _at(30, 39.5), _at(40.5, 42)]
    assert chunker._windows(ranges, speech, _at(3)[0], _at(75)[0]) == [
        _at(0, 12), _at(0, 12), _at(30, 42), _at(30, 42)
    ]
    assert chunker._windows(ranges, speech, 0, _at(75)[0]) == ranges


V2 = {"target_sec": 25.0, "max_sec": 30.0, "min_sec": 20.0, "context_sec": 5.0}  # ranges of 20 s at most


def _words(seconds, quiet=(), word_db=-20.0, gap_db=-30.0, seed=0, word=0.3, gap=0.1):
    """Noise like speech: `word` s words with `gap` s gaps 10 dB down between
    them, and each (start, end) second span in `quiet` a gap 35 dB down."""
    rng = np.random.default_rng(seed)
    total = int(seconds * SR)
    level = np.full(total, 10 ** (word_db / 20), dtype=np.float32)
    phase = np.arange(total) % int((word + gap) * SR)
    level[phase >= int(word * SR)] = 10 ** (gap_db / 20)
    for start, end in quiet:
        level[_at(start)[0]: _at(end)[0]] = 10 ** ((word_db - 35) / 20)
    return rng.standard_normal(total).astype(np.float32) * level


def _forced(monkeypatch, wav, speech, **bounds):
    monkeypatch.setattr(chunker, "_speech_segments", lambda _wav: [_at(*span) for span in speech])
    return chunker.plan_chunks(wav, **bounds)


def test_a_cut_in_speech_lands_in_the_quietest_gap_near_it(monkeypatch):
    # 41 s with no pause is three pieces of 20 s at most; split evenly, the
    # cuts fall at 13.67 s and 27.33 s, mid-word. Each moves into the deep
    # gap 2 s from it instead, past the shallower gaps between nearer words.
    quiet = [(11.7, 11.85), (29.3, 29.45)]
    ranges = _forced(monkeypatch, _words(41, quiet), [(0, 41)], **V2).ranges
    assert len(ranges) == 3 and ranges[0][0] == 0 and ranges[-1][1] == _at(41)[0]
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    for (_start, cut), (low, high) in zip(ranges, quiet):
        assert _at(low)[0] < cut < _at(high)[0]
    _assert_valid(ranges, _at(41)[0], _at(20)[0])


def test_a_cut_in_speech_without_a_deep_gap_still_lands_between_words(monkeypatch):
    # gaps of 0.25 s 10 dB down every 0.6 s: each cut goes in one near the even split
    ranges = _forced(monkeypatch, _words(41, word=0.35, gap=0.25), [(0, 41)], **V2).ranges
    assert len(ranges) == 3
    for index, (_start, cut) in enumerate(ranges[:-1], 1):
        assert cut % _at(0.6)[0] > _at(0.35)[0]
        assert abs(cut - _at(41)[0] * index // 3) < SR


def test_a_cut_in_speech_stays_put_where_no_point_near_is_clearly_quieter(monkeypatch):
    # gaps of 0.1 s 10 dB down: no 200 ms near the even split is 3 dB under it
    ranges = _forced(monkeypatch, _words(41), [(0, 41)], **V2).ranges
    assert ranges == chunker._split_oversized(0, _at(41)[0], _at(20)[0])


@pytest.mark.parametrize(
    ("dip", "db"),
    [
        ((14.9, 15.0), -35),  # a stop closure inside the word at 14.8-15.1 s
        ((14.8, 15.1), -16),  # that word said 16 dB quieter than the rest
    ],
)
def test_a_cut_already_in_a_gap_does_not_move_into_a_word(monkeypatch, dip, db):
    # 41.85 s splits evenly at 13.95 s, in the 0.1 s gap at 13.9-14.0 s. The
    # dip inside a word 1 s on is quieter than that gap, but no gap.
    wav = _words(41.85)
    wav[slice(*_at(*dip))] *= 10 ** (db / 20)
    ranges = _forced(monkeypatch, wav, [(0, 41.85)], **V2).ranges
    assert ranges == chunker._split_oversized(0, _at(41.85)[0], _at(20)[0])
    assert _at(13.9)[0] < ranges[0][1] < _at(14.0)[0]


def test_a_dip_shorter_than_a_syllable_does_not_draw_the_cut(monkeypatch):
    # 20 ms of silence mid-word at the even split, and a 150 ms gap 1 s off
    wav = _words(41, gap_db=-20.0)  # no gaps between words
    wav[_at(13.66)[0]: _at(13.68)[0]] = 0
    wav[_at(14.6)[0]: _at(14.75)[0]] *= 10 ** (-35 / 20)
    cut = _forced(monkeypatch, wav, [(0, 41)], **V2).ranges[0][1]
    assert _at(14.6)[0] < cut < _at(14.75)[0]


def test_speech_as_loud_throughout_keeps_the_even_split(monkeypatch):
    # nothing 6 dB under the rest: no cut wanders after noise in the level
    wav = _words(41, gap_db=-20.0)
    ranges = _forced(monkeypatch, wav, [(0, 41)], **V2).ranges
    assert ranges == chunker._split_oversized(0, _at(41)[0], _at(20)[0])


def test_a_cut_in_speech_reaches_a_gap_past_the_even_split_by_giving_up_silence(monkeypatch):
    # 38.5 s of speech spans 40 s with the margin after it: two pieces of
    # exactly 20 s, the cut fixed at 20 s. Ending the last range sooner, in
    # the margin's silence, lets the cut move to the gap at 19.6 s.
    wav = _words(45, [(19.6, 19.75)])
    wav[_at(38.5)[0]:] *= 1e-3
    ranges = _forced(monkeypatch, wav, [(0, 38.5)], **V2).ranges
    assert len(ranges) == 2 and ranges[0][0] == 0 and ranges[0][1] == ranges[1][0]
    assert _at(19.6)[0] < ranges[0][1] < _at(19.75)[0]
    assert _at(38.5)[0] <= ranges[1][1] <= _at(40)[0]
    _assert_valid(ranges, _at(45)[0], _at(20)[0])


def _margin_word(quiet, word):
    """45 s: words from 3.15 s to 37.3 s (all VAD hears), the gap `quiet`
    among them 35 dB down, room tone either side, and if `word`, a word VAD
    missed there at -42 dBFS, 22 dB under the rest."""
    wav = _words(45, [quiet])
    wav[: _at(3.15)[0]] *= 10 ** (-55 / 20)
    wav[_at(37.3)[0]:] *= 10 ** (-55 / 20)
    if word:
        span = slice(*_at(*word))
        wav[span] = np.random.default_rng(5).standard_normal(span.stop - span.start) * 10 ** (-42 / 20)
    return wav


@pytest.mark.parametrize(
    ("quiet", "word"),
    [
        ((18.4, 18.55), (38.3, 38.7)),  # the cut reaches the gap only if the range ends by 38.55 s
        ((21.7, 21.85), (1.4, 1.8)),  # ... only if it starts from 1.7 s
    ],
)
def test_a_split_first_or_last_range_keeps_a_quiet_word_in_its_margin(monkeypatch, quiet, word):
    # The first and last ranges reach up to CHUNK_TRIM_SILENCE_SEC past the
    # speech, in case VAD missed a quiet syllable there; nothing else decodes
    # that audio. A cut may have that margin's silence, never a sound in it.
    plan = _forced(monkeypatch, _margin_word(quiet, None), [(3.15, 37.3)], **V2)
    assert _at(quiet[0])[0] < plan.ranges[0][1] < _at(quiet[1])[0]  # silence alone gives way
    plan = _forced(monkeypatch, _margin_word(quiet, word), [(3.15, 37.3)], **V2)
    low, high = _at(*word)
    assert len(plan.ranges) == 2
    assert plan.ranges[0][0] <= low and high <= plan.ranges[-1][1]
    assert plan.windows[0][0] <= low and high <= plan.windows[-1][1]
    _assert_valid(plan.ranges, _at(45)[0], _at(20)[0])


@pytest.mark.parametrize(
    ("quiet", "margin"),
    [
        ((18.4, 18.55), (37.3, 45)),  # as above, the tail
        ((21.7, 21.85), (0, 3.15)),  # the lead
    ],
)
def test_a_split_first_or_last_range_keeps_a_quieter_talker_in_its_margin(monkeypatch, quiet, margin):
    # A margin holding no room tone, only a talker 22 dB under the rest: its
    # quietest tenth is that talker's own gaps, which its words don't clear
    # by 10 dB. It is all sound, and the ranges keep as much of it as an
    # even split's.
    wav = _margin_word(quiet, None)
    span = slice(*_at(*margin))
    wav[span] = _words(45, word_db=-42.0, gap_db=-48.0, seed=7)[span]
    plan = _forced(monkeypatch, wav, [(3.15, 37.3)], **V2)
    monkeypatch.setattr(chunker, "_quiet_cuts", lambda _wav, even, *_edges: even)
    even = _forced(monkeypatch, wav, [(3.15, 37.3)], **V2).ranges
    assert len(plan.ranges) == len(even) == 2
    if margin[0]:
        assert even[-1][1] <= plan.ranges[-1][1]
    else:
        assert plan.ranges[0][0] <= even[0][0]


@pytest.mark.parametrize("bed", ["hum", "offset"])
@pytest.mark.parametrize(
    ("quiet", "word"),
    [
        ((18.4, 18.55), (38.5, 39.0)),
        ((21.7, 21.85), (1.5, 2.0)),
    ],
)
def test_a_split_first_or_last_range_keeps_a_word_under_a_bed_in_its_margin(monkeypatch, quiet, word, bed):
    # A 50 Hz hum at -40 dBFS, or a DC offset as loud, under the whole file:
    # the margin's floor is the bed, and a -42 dBFS word VAD missed there
    # is under it, not 10 dB over it. It is sound all the same.
    wav = _margin_word(quiet, word)
    t = np.arange(wav.size) / SR
    wav += (np.sin(2 * np.pi * 50 * t) * np.sqrt(2) if bed == "hum" else np.ones_like(t)).astype(np.float32) * 0.01
    plan = _forced(monkeypatch, wav, [(3.15, 37.3)], **V2)
    low, high = _at(*word)
    assert len(plan.ranges) == 2
    assert plan.ranges[0][0] <= low and high <= plan.ranges[-1][1]


def test_a_cut_in_speech_may_give_the_pause_after_it_to_the_next_range(monkeypatch):
    # The first phrase's range is cut at 40 s, early in the pause, to stay
    # two pieces; ending it at the pause's start instead gives the next range
    # 0.4 s more, which still fits one piece, and lets its cut reach 19.7 s.
    wav = _words(60, [(19.7, 19.85)])
    wav[_at(39.6)[0]: _at(41)[0]] *= 1e-3
    speech = [(0, 39.6), (41, 50)]
    ranges = _forced(monkeypatch, wav, speech, **V2).ranges
    assert len(ranges) == 3
    assert _at(19.7)[0] < ranges[0][1] < _at(19.85)[0]
    assert _at(39.6)[0] <= ranges[1][1] <= _at(40)[0] and ranges[2][0] == ranges[1][1]
    _assert_valid(ranges, _at(60)[0], _at(20)[0])


@pytest.mark.parametrize(
    "bounds",
    [V2, {**V2, "context_sec": 0.0}, {**BOUNDS, "context_sec": 5.0}],
)
def test_quiet_cuts_keep_every_bound_and_add_no_piece(monkeypatch, bounds):
    # Random layouts of long and short phrases, with words in them: the
    # pieces are those of an even split, all fit, and none is a sliver.
    rng = np.random.default_rng(3)
    own_maximum = _at(bounds["max_sec"] - 2 * bounds["context_sec"])[0]
    for seed in range(8):
        wav = _words(200, seed=seed) * rng.uniform(0.2, 1, 1).astype(np.float32)
        speech, at = [], rng.uniform(0, 2)
        while at < 190:
            length = rng.uniform(20, 90) if rng.random() < 0.4 else rng.uniform(1, 6)
            speech.append((at, min(199, at + length)))
            at += length + rng.uniform(0.45, 4)
        for start, end in zip([s[1] for s in speech], [s[0] for s in speech[1:]] + [200]):
            wav[_at(start)[0]: _at(end)[0]] *= 1e-3
        plan = _forced(monkeypatch, wav, speech, **bounds)
        monkeypatch.setattr(chunker, "_quiet_cuts", lambda _wav, even, *_edges: even)
        even = _forced(monkeypatch, wav, speech, **bounds).ranges
        monkeypatch.undo()
        ranges, windows = plan.ranges, plan.windows
        assert len(ranges) == len(even)
        _assert_valid(ranges, wav.size, own_maximum)
        for (start, end), (window_start, window_end) in zip(ranges, windows):
            assert window_start <= start < end <= window_end
            assert window_end - window_start <= bounds["max_sec"] * SR
        # nothing VAD heard is outside every range
        for start, end in speech:
            covered = sum(max(0, min(_at(end)[0], b) - max(_at(start)[0], a)) for a, b in ranges)
            assert covered == _at(end)[0] - _at(start)[0]
        # no piece shorter than a quarter of the maximum, where an even split had none
        assert min(b - a for a, b in ranges) >= min(min(b - a for a, b in even), own_maximum // 4)


def test_slice_chunks_returns_views():
    waveform = np.arange(20, dtype=np.float32)
    pieces = chunker.slice_chunks(waveform, [(2, 8), (8, 12)])
    assert len(pieces) == 2
    assert np.shares_memory(waveform, pieces[0])
    assert pieces[0].flags.c_contiguous


def test_vad_boundaries_do_not_trim_quiet_first_or_last_words(monkeypatch):
    sr = chunker.TARGET_SR
    total = int(MAX_SEC * 3 * sr)
    margin = sr  # shorter than CHUNK_TRIM_SILENCE_SEC: kept
    monkeypatch.setattr(chunker, "_speech_segments", lambda _wav: [(margin, total - margin)])
    ranges = _ranges(np.ones(total, dtype=np.float32), **BOUNDS)
    _assert_valid(ranges, total, int(MAX_SEC * sr))
    assert ranges[0][0] == 0
    assert ranges[-1][1] == total
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))


def test_long_edge_silence_is_cut_but_keeps_a_margin(monkeypatch):
    sr = chunker.TARGET_SR
    total = int(MAX_SEC * 3 * sr)
    margin = 10 * sr  # longer than CHUNK_TRIM_SILENCE_SEC: mostly cut
    trim = int(chunker.CHUNK_TRIM_SILENCE_SEC * sr)
    monkeypatch.setattr(chunker, "_speech_segments", lambda _wav: [(margin, total - margin)])
    ranges = _ranges(np.ones(total, dtype=np.float32), **BOUNDS)
    _assert_valid(ranges, total, int(MAX_SEC * sr))
    assert ranges[0][0] == margin - trim
    assert ranges[-1][1] == total - margin + trim


def test_each_thread_has_its_own_vad(monkeypatch):
    # Silero is stateful: sharing one model would serialize every request's VAD
    monkeypatch.setattr(chunker, "_vad_local", threading.local())
    monkeypatch.setattr(chunker, "_load_vad", object)
    here = chunker._get_vad()
    assert chunker._get_vad() is here
    with ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(chunker._get_vad).result() is not here


SR = chunker.TARGET_SR


def _bursts(level_db, gap_db, bursts=3, burst_sec=1.0, gap_sec=1.0):
    """Tone bursts at level_db, with gaps of noise at gap_db, both in dBFS RMS."""
    rng = np.random.default_rng(0)
    tone = np.sin(2 * np.pi * 220 * np.arange(int(burst_sec * SR)) / SR) * np.sqrt(2) * 10 ** (level_db / 20)
    gap = rng.standard_normal(int(gap_sec * SR)) * 10 ** (gap_db / 20)
    return np.concatenate([part for _ in range(bursts) for part in (gap, tone)] + [gap]).astype(np.float32)


@pytest.fixture
def volume(monkeypatch):
    def fail():
        raise AssertionError("PARAKEET_VAD=volume must not load Silero")

    monkeypatch.setattr(chunker, "VAD", "volume")
    monkeypatch.setattr(chunker, "VAD_GATE_DB", None)
    monkeypatch.setattr(chunker, "_get_vad", fail)


def test_volume_finds_speech_without_silero(volume):
    segments = chunker._speech_segments(_bursts(-25, -70))
    pad = int(chunker.VAD_SPEECH_PAD_MS * SR / 1000)
    # each 1 s burst, from its 20 ms frames, padded as Silero pads
    assert [(round(a / SR, 2), round(b / SR, 2)) for a, b in segments] == [
        (round(1 - pad / SR, 2), round(2 + pad / SR, 2)),
        (round(3 - pad / SR, 2), round(4 + pad / SR, 2)),
        (round(5 - pad / SR, 2), round(6 + pad / SR, 2)),
    ]


def test_volume_gate_follows_the_file_unless_fixed(volume, monkeypatch):
    quiet = _bursts(-45, -80)  # a quiet recording: the file's own gate still finds it
    assert len(chunker._speech_segments(quiet)) == 3
    monkeypatch.setattr(chunker, "VAD_GATE_DB", -40.0)  # a fixed gate above it hears nothing
    assert chunker._speech_segments(quiet) == []
    monkeypatch.setattr(chunker, "VAD_GATE_DB", -60.0)
    assert len(chunker._speech_segments(quiet)) == 3


def test_volume_joins_speech_across_dips_shorter_than_a_pause(volume, monkeypatch):
    monkeypatch.setattr(chunker, "VAD_MIN_SILENCE_MS", 400)
    assert len(chunker._speech_segments(_bursts(-25, -70, gap_sec=0.38))) == 1
    assert len(chunker._speech_segments(_bursts(-25, -70, gap_sec=0.4))) == 3


def test_volume_hears_no_pause_above_the_gate(volume, monkeypatch):
    # noise louder than a fixed gate fills the pauses: one span, so long audio is cut by length
    monkeypatch.setattr(chunker, "VAD_GATE_DB", -50.0)
    assert len(chunker._speech_segments(_bursts(-25, -40))) == 1


def _turn(seconds, level_db, floor_db=-70):
    """A speaker's turn: 250 ms syllables at level_db, with 100 ms dips to
    floor_db between them."""
    return _bursts(level_db, floor_db, bursts=round(seconds / 0.35), burst_sec=0.25, gap_sec=0.1)


@pytest.mark.parametrize("quiet_db", [-34, -44])
def test_volume_keeps_a_quieter_speakers_turn(volume, quiet_db):
    # 50 s at -20 dBFS, a second speaker's 10 s 14 or 24 dB quieter, 50 s at
    # -20 dBFS. Under 0.4x the file's average throughout, the quiet turn was
    # one long pause, cut out of the chunks and never decoded.
    first, quiet, last = _turn(50, -20), _turn(10, quiet_db), _turn(50, -20)
    plan = chunker.plan_chunks(np.concatenate([first, quiet, last]), **BOUNDS, context_sec=5.0)
    start, end = first.size, first.size + quiet.size
    assert sum(max(0, min(b, end) - max(a, start)) for a, b in plan.ranges) == quiet.size


def test_volume_still_cuts_out_a_long_pause_of_room_tone(volume):
    # Heard again at its own level, room tone louder than -60 dBFS never rises
    # 10 dB over its own floor: still a pause, cut out.
    first, last = _turn(40, -20, floor_db=-50), _turn(40, -20, floor_db=-50)
    tone = (np.random.default_rng(1).standard_normal(10 * SR) * 10 ** (-50 / 20)).astype(np.float32)
    ranges = chunker.plan_chunks(np.concatenate([first, tone, last]), **BOUNDS, context_sec=5.0).ranges
    assert [(round(a / SR), round(b / SR)) for a, b in ranges] == [(0, 40), (50, 90)]


def _pause_between_turns(pause_sec, sounds):
    """40 s turns at -20 dBFS either side of a pause of -55 dBFS room tone,
    with `sounds` in it: (seconds into the pause, seconds long, dBFS)."""
    rng = np.random.default_rng(1)
    pause = rng.standard_normal(int(pause_sec * SR)) * 10 ** (-55 / 20)
    for at, seconds, level_db in sounds:
        n = int(seconds * SR)
        pause[int(at * SR): int(at * SR) + n] += rng.standard_normal(n) * 10 ** (level_db / 20)
    first, last = _turn(40, -20, floor_db=-55), _turn(40, -20, floor_db=-55)
    return np.concatenate([first, pause.astype(np.float32), last])


@pytest.mark.parametrize(
    ("pause_sec", "sounds"),
    [
        (10, [(4.9, 0.01, -30)]),  # a click: a frame ~22 dB over the room tone
        (10, [(4.9, 0.03, -35)]),  # a knock: two frames 17-20 dB over it
        (10, [(1.0 + 0.27 * k, 0.01, -30) for k in range(30)]),  # thirty clicks: a mean of 100 ms passed them
        (5, [(2.4, 0.3, -42)]),  # a breath: 13 dB over it, for 300 ms
        (15, [(3.5, 0.25, -42), (8.0, 0.25, -42), (12.5, 0.25, -42)]),  # three: 0.75 s summed, under 0.5 s each
        (10, [(4.6, 0.17, -42), (5.07, 0.17, -42)]),  # a rustle: 0.64 s from end to end, 0.34 s of it loud
        (10, [(4.8, 0.45, -40)]),  # 450 ms 15 dB over it: as long as a breath, under half a second
        (10, [(4.5, 0.3, -42), (5.25, 0.3, -42)]),  # two breaths 0.45 s apart: two sounds, not one of 0.6 s
    ],
)
def test_volume_still_cuts_out_a_long_pause_with_sounds_in_it(volume, pause_sec, sounds):
    # No syllable, or no sound long enough to be speech: the pause is still
    # cut out whole, not decoded as chunks of their own around the sounds,
    # where Parakeet can make up a word.
    wav = _pause_between_turns(pause_sec, sounds)
    ranges = chunker.plan_chunks(wav, **BOUNDS, context_sec=5.0).ranges
    assert [(round(a / SR), round(b / SR)) for a, b in ranges] == [(0, 40), (40 + pause_sec, 80 + pause_sec)]


def test_volume_keeps_a_quieter_speakers_short_words_between_phrases(volume):
    # Two 2 s phrases 24 dB under the rest, and between them 4 s of short
    # words alone (250 ms, 0.8 s apart): each under half a second, but near
    # speech. Dropped, they left 4 s to cut out of the quiet turn.
    words = np.concatenate([np.concatenate([part, np.zeros(int(0.55 * SR), np.float32)])
                            for part in [_turn(0.25, -44)[: SR // 4]] * 5])
    quiet = np.concatenate([_turn(2, -44), words, _turn(2, -44)])
    quiet += (np.random.default_rng(2).standard_normal(quiet.size) * 10 ** (-70 / 20)).astype(np.float32)
    first, last = _turn(50, -20), _turn(50, -20)
    plan = chunker.plan_chunks(np.concatenate([first, quiet, last]), **BOUNDS, context_sec=5.0)
    start, end = first.size, first.size + quiet.size
    assert sum(max(0, min(b, end) - max(a, start)) for a, b in plan.ranges) == quiet.size


@pytest.mark.parametrize(
    "sounds",
    [
        [(4.8, 0.55, -40)],  # 550 ms 15 dB over the room tone
        [(4.8, 0.3, -40), (5.4, 0.3, -40)],  # two 0.3 s 0.3 s apart: one sound, 0.6 s of it loud
    ],
)
def test_volume_keeps_a_quiet_reply_over_half_a_second(volume, sounds):
    # Alone in a 10 s pause: decoded.
    wav = _pause_between_turns(10, sounds)
    ranges = chunker.plan_chunks(wav, **BOUNDS, context_sec=5.0).ranges
    reply = (int((40 + sounds[0][0]) * SR), int((40 + sum(sounds[-1][:2])) * SR))
    assert any(a <= reply[0] and reply[1] <= b for a, b in ranges)
    assert [(round(a / SR), round(b / SR)) for a, b in ranges if b < reply[0] or a > reply[1]] == [(0, 40), (50, 90)]


@pytest.mark.parametrize("word", [(8.5, 0.3), (8.95, 0.4)])
def test_volume_keeps_a_quiet_word_near_a_phrase_whole(volume, word):
    # A quiet 1 s phrase, then a word 2.5 s after it, or one reaching past the
    # 3 s around it, then a pause: the whole word is decoded. Kept frame by
    # frame, the second was cut at 3 s, 0.23 s of it never decoded.
    wav = _pause_between_turns(20, [(5.0, 1.0, -40), (*word, -40)])
    start, end = int((40 + word[0]) * SR), int((40 + sum(word)) * SR)
    assert any(a <= start and end <= b for a, b in chunker.plan_chunks(wav, **BOUNDS).ranges)


def test_volume_keeps_no_train_of_clicks_far_past_a_phrase(volume):
    # Seven clicks 0.36 s apart, one sound, from just inside the 3 s past a
    # quiet phrase: kept no more than half a second past it, so the rest of
    # the pause is still cut out. Kept whole, the clicks left too little to.
    clicks = [(4.8 + 0.36 * k, 0.06, -44) for k in range(7)]
    wav = _pause_between_turns(10, [(1.0, 1.0, -40)] + clicks)
    ranges = chunker.plan_chunks(wav, **BOUNDS, context_sec=5.0).ranges
    assert [(round(a / SR), round(b / SR)) for a, b in ranges] == [(0, 45), (50, 90)]


def test_volume_hears_quiet_sounds_whatever_the_shortest_pause(volume, monkeypatch):
    # Joined across PARAKEET_VAD_MIN_SILENCE_MS alone, at 60 ms a quiet turn's
    # syllables were each a sound too short to be speech, and so were two
    # 0.3 s words 0.3 s apart. Still 400 ms.
    monkeypatch.setattr(chunker, "VAD_MIN_SILENCE_MS", 60)
    first, quiet, last = _turn(50, -20), _turn(10, -44), _turn(50, -20)
    plan = chunker.plan_chunks(np.concatenate([first, quiet, last]), **BOUNDS, context_sec=5.0)
    start, end = first.size, first.size + quiet.size
    assert sum(max(0, min(b, end) - max(a, start)) for a, b in plan.ranges) == quiet.size
    wav = _pause_between_turns(10, [(4.8, 0.3, -40), (5.4, 0.3, -40)])
    ranges = chunker.plan_chunks(wav, **BOUNDS, context_sec=5.0).ranges
    assert any(a <= int(44.8 * SR) and int(45.7 * SR) <= b for a, b in ranges)


def test_volume_joins_quiet_words_across_a_longer_shortest_pause(volume, monkeypatch):
    # Ten 0.35 s words 0.5 s apart, alone in a pause: each under half a second,
    # but one sound across a PARAKEET_VAD_MIN_SILENCE_MS of 1 s, decoded.
    # Joined across 400 ms alone, they were dropped.
    monkeypatch.setattr(chunker, "VAD_MIN_SILENCE_MS", 1000)
    wav = _pause_between_turns(12, [(1.0 + 0.85 * k, 0.35, -42) for k in range(10)])
    start, end = int(41.0 * SR), int(48.7 * SR)
    ranges = chunker.plan_chunks(wav, **BOUNDS, context_sec=5.0).ranges
    assert sum(max(0, min(b, end) - max(a, start)) for a, b in ranges) == end - start


def test_a_fixed_gate_is_not_heard_again(volume, monkeypatch):
    # The operator's gate: all under it is silence, a quiet speaker too.
    monkeypatch.setattr(chunker, "VAD_GATE_DB", -30.0)
    first, quiet, last = _turn(50, -20), _turn(10, -40), _turn(50, -20)
    plan = chunker.plan_chunks(np.concatenate([first, quiet, last]), **BOUNDS, context_sec=5.0)
    start, end = first.size, first.size + quiet.size
    assert sum(max(0, min(b, end) - max(a, start)) for a, b in plan.ranges) < quiet.size


def test_silero_without_the_package_falls_back_to_volume(monkeypatch):
    monkeypatch.setattr(chunker, "VAD", "silero")
    monkeypatch.setattr(chunker, "VAD_GATE_DB", None)
    monkeypatch.setattr(chunker, "_get_vad", lambda: "energy")
    assert len(chunker._speech_segments(_bursts(-25, -70))) == 3


@pytest.mark.parametrize(("raw", "expected"), [(None, None), ("", None), (" -45 ", -45.0), ("-50.5", -50.5)])
def test_gate_setting_parses(monkeypatch, raw, expected):
    from parakeet_service import config

    if raw is None:
        monkeypatch.delenv("PARAKEET_VAD_GATE_DB", raising=False)
    else:
        monkeypatch.setenv("PARAKEET_VAD_GATE_DB", raw)
    assert config._env_dbfs("PARAKEET_VAD_GATE_DB") == expected


@pytest.mark.parametrize("raw", ["loud", "0", "6", "-121"])
def test_gate_setting_rejects_nonsense(monkeypatch, raw):
    from parakeet_service import config

    monkeypatch.setenv("PARAKEET_VAD_GATE_DB", raw)
    with pytest.raises(RuntimeError, match="PARAKEET_VAD_GATE_DB"):
        config._env_dbfs("PARAKEET_VAD_GATE_DB")


def test_frame_rms_in_blocks_matches_one_pass(monkeypatch):
    rng = np.random.default_rng(1)
    wav = (0.1 * rng.standard_normal(SR * 3 + 123)).astype(np.float32)
    count = wav.size // chunker.FRAME
    framed = wav[: count * chunker.FRAME].reshape(count, chunker.FRAME)
    whole = np.sqrt((framed * framed).mean(axis=1) + 1e-12)
    monkeypatch.setattr(chunker, "_RMS_BLOCK", 7)  # many blocks, one partial
    assert np.array_equal(chunker.frame_rms(wav), whole)
    assert chunker.frame_rms(np.zeros(10, dtype=np.float32)).size == 0
