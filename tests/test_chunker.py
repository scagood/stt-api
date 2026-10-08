from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from parakeet_service import chunker, routes

MAX_SEC = 75.0
BOUNDS = {"target_sec": 60.0, "max_sec": MAX_SEC}


def _assert_valid(ranges, total, maximum):
    previous_end = -1
    for start, end in ranges:
        assert 0 <= start < end <= total
        assert end - start <= maximum
        assert start >= previous_end
        previous_end = end


def test_empty_audio_has_no_chunks():
    assert chunker.auto_chunk(np.empty(0, dtype=np.float32), **BOUNDS) == []


def test_short_audio_bypasses_vad(monkeypatch):
    monkeypatch.setattr(
        chunker,
        "_speech_segments",
        lambda _wav: (_ for _ in ()).throw(AssertionError("VAD should not run")),
    )
    waveform = np.zeros(int(MAX_SEC * chunker.TARGET_SR) - 1)
    assert chunker.auto_chunk(waveform, **BOUNDS) == [(0, waveform.size)]


def test_long_silence_skips_inference(monkeypatch):
    monkeypatch.setattr(chunker, "_speech_segments", lambda _wav: [])
    waveform = np.zeros(int((MAX_SEC + 10) * chunker.TARGET_SR))
    assert chunker.auto_chunk(waveform, **BOUNDS) == []


def test_long_uninterrupted_speech_has_no_phantom_tail(monkeypatch):
    total = int((MAX_SEC * 2.5) * chunker.TARGET_SR)
    monkeypatch.setattr(
        chunker, "_speech_segments", lambda _wav: [(0, total)]
    )
    ranges = chunker.auto_chunk(np.ones(total, dtype=np.float32), **BOUNDS)
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
    ranges = chunker.auto_chunk(np.ones(total, dtype=np.float32), **BOUNDS)
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
    ranges = chunker.auto_chunk(
        np.ones(total, dtype=np.float32), target_sec=25.0, max_sec=30.0, min_sec=20.0
    )
    _assert_valid(ranges, total, int(30.0 * sr))
    assert ranges[0][0] == 0
    assert ranges[-1][1] == total


def test_short_audio_is_one_piece_whatever_the_context():
    waveform = np.zeros(int(MAX_SEC * chunker.TARGET_SR))
    assert chunker.auto_chunk(waveform, **BOUNDS, context_sec=5.0) == [(0, waveform.size)]


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
    ranges = chunker.auto_chunk(
        np.broadcast_to(np.float32(0), (total,)),  # hours of samples in no memory
        target_sec=target_sec, max_sec=max_sec, min_sec=min_sec, context_sec=context_sec,
    )
    assert len(ranges) <= most
    _assert_valid(ranges, total, own_maximum)
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    cuts = [end for _start, end in ranges[:-1]]
    assert not any(start < cut < end for cut in cuts for start, end in speech)  # all in pauses
    # every range holds speech: no slivers of silence
    assert all(any(begin < end and start < stop for begin, stop in speech) for start, end in ranges)
    # each closes at the last pause before its target: short of it by less than a phrase and a pause
    assert all(end - start > own_target - 7.5 * SR for start, end in ranges[:-1])


def test_ranges_close_in_a_pause_rather_than_pass_their_maximum(monkeypatch):
    # parakeet-v2 with 5 s of context: ranges of 20 s at most, target 20 s,
    # minimum 20 s. The first range is 19.8 s at the pause, short of the
    # minimum, but taking the next phrase would pass 20 s: it closes in the
    # pause, at 20 s rather than mid-way at 20.2 s, and the last range's
    # margin past the speech stops at 20 s too. Neither leaves a sliver.
    speech = [_at(3, 19.8), _at(20.6, 39.5)]
    monkeypatch.setattr(chunker, "_speech_segments", lambda _wav: speech)
    ranges = chunker.auto_chunk(
        np.zeros(_at(45)[0], dtype=np.float32), target_sec=25.0, max_sec=30.0, min_sec=20.0, context_sec=5.0
    )
    assert ranges == [_at(0, 20), _at(20, 40)]


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
    ranges = chunker.auto_chunk(np.ones(total, dtype=np.float32), **BOUNDS)
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
    ranges = chunker.auto_chunk(np.ones(total, dtype=np.float32), **BOUNDS)
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


def test_volume_hears_no_pause_above_the_gate(volume, monkeypatch):
    # noise louder than a fixed gate fills the pauses: one span, so long audio is cut by length
    monkeypatch.setattr(chunker, "VAD_GATE_DB", -50.0)
    assert len(chunker._speech_segments(_bursts(-25, -40))) == 1


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
