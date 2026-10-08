from __future__ import annotations

import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from parakeet_service import routes
from parakeet_service.config import TARGET_SR

# The aligner a request names (option C: there is no default).
EAR = ("wav2vec2-base-960h", "int8")


def _samples(ranges_sec):
    return [(int(s * TARGET_SR), int(e * TARGET_SR)) for s, e in ranges_sec]


def _prepared(ranges_sec, pieces=None, windows_sec=None):
    ranges = _samples(ranges_sec)
    windows = _samples(windows_sec) if windows_sec else ranges
    duration = max(e for _s, e in ranges_sec)
    return routes._PreparedAudio(
        waveform=None,
        ranges=ranges,
        windows=windows,
        speech=[],
        pieces=pieces or [np.zeros(end - start, dtype=np.float32) for start, end in windows],
        duration=duration,
    )


def _result(tokens, timestamps):
    text = "".join(t.replace("▁", " ") for t in tokens).strip()
    return SimpleNamespace(text=text, tokens=tokens, timestamps=timestamps)


def test_word_end_does_not_absorb_pause():
    result = _result(
        [" Hello", " wor", "ld", ".", " Then"],
        [0.0, 0.5, 0.7, 0.9, 21.0],  # 20 s pause inside the chunk
    )
    _text, _segments, words = routes._stitch(_prepared([(0.0, 30.0)]), [result])
    assert [w["word"] for w in words] == ["Hello", "world.", "Then"]
    pre_pause = words[1]
    assert pre_pause["end"] <= 0.9 + routes._WORD_TAIL_SEC + 1e-9
    assert all(w["end"] - w["start"] < 1.5 for w in words)


def test_last_word_does_not_balloon_to_chunk_end():
    result = _result([" Deep", " breath", "."], [0.0, 0.4, 0.8])
    _text, _segments, words = routes._stitch(_prepared([(0.0, 75.0)]), [result])
    assert words[-1]["end"] <= 0.8 + routes._WORD_TAIL_SEC + 1e-9


def test_token_timestamp_mismatch_drops_no_words():
    result = _result([" one", " two", " three", " four"], [0.0, 0.5])
    _text, _segments, words = routes._stitch(_prepared([(0.0, 10.0)]), [result])
    assert [w["word"] for w in words] == ["one", "two", "three", "four"]
    starts = [w["start"] for w in words]
    assert starts == sorted(starts)


def test_invalid_timestamps_reuse_previous_and_drop_nothing():
    result = _result(
        [" one", " two", " three", " four"],
        [0.0, float("nan"), float("inf"), -5.0],
    )
    _text, _segments, words = routes._stitch(_prepared([(0.0, 10.0)]), [result])
    assert [w["word"] for w in words] == ["one", "two", "three", "four"]
    starts = [w["start"] for w in words]
    assert starts == sorted(starts)
    assert all(w["start"] >= 0.0 for w in words)


def test_bpe_pieces_group_into_one_word():
    result = _result(["▁lig", "ht", "ho", "use"], [0.0, 0.1, 0.2, 0.3])
    _text, _segments, words = routes._stitch(_prepared([(0.0, 5.0)]), [result])
    assert [w["word"] for w in words] == ["lighthouse"]
    assert words[0]["start"] == 0.0
    assert words[0]["end"] <= 0.3 + routes._WORD_TAIL_SEC + 1e-9


def test_second_chunk_words_use_chunk_offset():
    first = _result([" one", "."], [0.0, 0.3])
    second = _result([" two", "."], [0.5, 0.8])
    _text, _segments, words = routes._stitch(
        _prepared([(0.0, 10.0), (10.0, 20.0)]), [first, second]
    )
    assert [w["word"] for w in words] == ["one.", "two."]
    assert words[1]["start"] == 10.5


def test_lone_word_marker_before_digits_and_currency_starts_a_word():
    # Parakeet v3 tokens, verbatim: the marker comes alone before "£" and digits,
    # and onnx_asr's text join has already dropped the space before the "£".
    result = SimpleNamespace(
        text="The coffee was£1.10 in 2005.",
        tokens=[" The", " co", "ff", "ee", " was", " ", "£", "1", ".", "1", "0", " in", " ", "2", "0", "0", "5", "."],
        timestamps=[0.1 * i for i in range(18)],
    )
    text, segments, words = routes._stitch(_prepared([(0.0, 5.0)]), [result])
    assert [w["word"] for w in words] == ["The", "coffee", "was", "£1.10", "in", "2005."]
    assert text == segments[0]["segment"] == "The coffee was £1.10 in 2005."


class _FakeChunk:
    """Stands in for aligner.ChunkAligner: re-times word i to (0.25 + i, 0.5 + i)."""

    def __init__(self, wav, calls):
        self.wav, self.calls = wav, calls

    def spans(self, words):
        self.calls.append((self.wav, list(words)))
        return [(0.25 + i, 0.5 + i) for i in range(len(words))]


def test_aligner_retimes_each_chunk_from_its_own_audio(monkeypatch):
    first = _result([" hi", " there"], [0.0, 0.8])
    second = _result([" bye"], [0.0])
    calls = []
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda wav, *_: _FakeChunk(wav, calls))

    _text, _segments, words = routes._stitch(
        _prepared([(0.0, 2.0), (5.0, 7.0)], pieces=["chunk0", "chunk1"]),
        [first, second],
        align=True, aligner_choice=EAR)
    assert calls == [("chunk0", ["hi", "there"]), ("chunk1", ["bye"])]
    assert [(w["word"], w["start"], w["end"]) for w in words] == [
        ("hi", 0.25, 0.5),
        ("there", 1.25, 1.5),
        ("bye", 5.25, 5.5),
    ]


def test_unavailable_aligner_keeps_model_times(monkeypatch):
    prepared = _prepared([(0.0, 2.0)])
    result = _result([" hi"], [0.4])
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda wav, *_: None)
    assert routes._stitch(prepared, [result], align=True, aligner_choice=EAR) == routes._stitch(prepared, [result])


def test_unaligned_words_stay_between_aligned_neighbours():
    words = [
        {"word": "a", "start": 0.0, "end": 0.3},
        {"word": "%", "start": 0.1, "end": 0.9},  # nothing the aligner can anchor
        {"word": "b", "start": 0.5, "end": 0.8},
    ]
    routes._apply_alignment(words, [(0.10, 0.20), None, (0.40, 0.60)], 10.0, 20.0)
    assert (words[0]["start"], words[0]["end"]) == (10.1, 10.2)
    assert (words[2]["start"], words[2]["end"]) == (10.4, 10.6)
    assert words[0]["end"] <= words[1]["start"] <= words[1]["end"] <= words[2]["start"]


def _talk_pause_talk():
    """Speech 0-2 s, a pause 2-3 s, speech 3-5 s (a tone stands in for speech)."""
    t = np.arange(5 * TARGET_SR) / TARGET_SR
    wav = 0.1 * np.sin(2 * np.pi * 220 * t)
    wav[2 * TARGET_SR: 3 * TARGET_SR] = 0.0
    return wav.astype(np.float32)


def _with_audio(ranges_sec, wav):
    prepared = _prepared(ranges_sec)
    prepared.waveform = wav
    return prepared


# Parakeet's times: "world." starts after the speech stopped, "Then" before it resumed.
LATE_AND_EARLY = ([" Hello", " world", ".", " Then", " more"], [0.4, 2.24, 2.4, 2.8, 3.6])


def test_retime_moves_words_out_of_pauses():
    prepared = _with_audio([(0.0, 5.0)], _talk_pause_talk())
    _text, segments, words = routes._stitch(prepared, [_result(*LATE_AND_EARLY)], retime_words=True)
    assert [(w["word"], round(w["start"], 2), round(w["end"], 2)) for w in words] == [
        ("Hello", 0.4, 0.72),
        ("world.", 1.52, 2.0),  # closes a sentence: before the pause, as long as it was
        ("Then", 3.0, 3.12),    # starts in the pause: at its end
        ("more", 3.6, 3.92),    # nowhere near one: untouched
    ]
    assert segments[0]["start"] <= words[0]["start"] and segments[0]["end"] >= words[-1]["end"]


def test_retime_is_off_unless_asked():
    prepared = _with_audio([(0.0, 5.0)], _talk_pause_talk())
    result = _result(*LATE_AND_EARLY)
    assert routes._stitch(prepared, [result]) == routes._stitch(_prepared([(0.0, 5.0)]), [result])
    assert routes._stitch(prepared, [result])[2][1]["start"] == 2.24


def test_aligned_words_are_not_retimed(monkeypatch):
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda wav, *_: _FakeChunk(wav, []))
    prepared = _with_audio([(0.0, 5.0)], _talk_pause_talk())
    _text, _segments, words = routes._stitch(
        prepared, [_result(*LATE_AND_EARLY)], align=True, aligner_choice=EAR, retime_words=True
    )
    assert [(w["start"], w["end"]) for w in words] == [(0.25 + i, 0.5 + i) for i in range(4)]


def _timed(words):
    return [(w["word"], round(w["start"], 3)) for w in words]


def test_each_piece_keeps_only_the_words_starting_in_its_own_range():
    # Two pieces cut at 10 s, each decoding 3 s of the other (plan_chunks).
    # The first hears the second's "Now, comrades" and makes up an "and" as its
    # input ends; the second hears the first's "there." (times from 7 s).
    first = _result(
        [" Hello", " there", ".", " Now", ",", " comrades", " and"],
        [1.0, 9.0, 9.3, 11.0, 11.3, 11.6, 12.8],
    )
    second = _result([" there", ".", " Now", ",", " comrades", "."], [2.0, 2.3, 4.0, 4.3, 4.6, 5.2])
    text, segments, words = routes._stitch(
        _prepared([(0.0, 10.0), (10.0, 20.0)], windows_sec=[(0.0, 13.0), (7.0, 20.0)]), [first, second]
    )
    assert text == "Hello there. Now, comrades."
    assert [s["segment"] for s in segments] == ["Hello there.", "Now, comrades."]
    assert _timed(words) == [("Hello", 1.0), ("there.", 9.0), ("Now,", 11.0), ("comrades.", 11.6)]
    assert round(segments[1]["start"], 3) == 11.0


def test_trimmed_text_is_rebuilt_as_onnx_asr_joins_it():
    # "So" is in the context before the range; the lone marker before "£" stays with its word
    result = SimpleNamespace(
        text="So it was£1.10.",
        tokens=[" So", " it", " was", " ", "£", "1", ".", "1", "0", "."],
        timestamps=[0.2, 2.5, 2.8, 3.0, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
    )
    text, _segments, words = routes._stitch(_prepared([(2.0, 6.0)], windows_sec=[(0.0, 6.0)]), [result])
    assert text == "it was £1.10."
    assert _timed(words) == [("it", 2.5), ("was", 2.8), ("£1.10.", 3.0)]


def test_aligner_hears_the_ranges_own_audio_not_its_context(monkeypatch):
    wav = np.arange(20 * TARGET_SR, dtype=np.float32)
    windows = [(0.0, 13.0), (7.0, 20.0)]
    pieces = [wav[start:end] for start, end in _samples(windows)]
    calls = []
    monkeypatch.setattr(routes.aligner, "for_chunk", lambda wav, *_: _FakeChunk(wav, calls))
    _text, _segments, words = routes._stitch(
        _prepared([(0.0, 10.0), (10.0, 20.0)], pieces=pieces, windows_sec=windows),
        [_result([" one"], [5.0]), _result([" two"], [4.0])],
        align=True, aligner_choice=EAR)
    assert [(heard[0], heard.size) for heard, _words in calls] == [(0, 10 * TARGET_SR), (10 * TARGET_SR, 10 * TARGET_SR)]
    assert _timed(words) == [("one", 0.25), ("two", 10.25)]


def _wav_bytes(wav):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(TARGET_SR)
        out.writeframes((wav * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def test_long_audio_decodes_each_range_with_its_context():
    # 200 s of 3.5 s tones and 0.5 s pauses: cut in pauses, about every 60 s
    tone = 0.3 * np.sin(2 * np.pi * 220 * np.arange(int(3.5 * TARGET_SR)) / TARGET_SR)
    wav = np.tile(np.concatenate([tone, np.zeros(TARGET_SR // 2)]), 50).astype(np.float32)
    prepared = routes._prepare_audio(_wav_bytes(wav), 60.0, 75.0, 20.0, 5.0)
    assert len(prepared.ranges) > 2 and prepared.speech
    assert all(end - start <= 65 * TARGET_SR for start, end in prepared.ranges)
    assert [piece.size for piece in prepared.pieces] == [end - start for start, end in prepared.windows]
    for (start, end), (window_start, window_end) in zip(prepared.ranges[:-1], prepared.windows[:-1]):
        assert window_start <= start and window_end >= end + 5 * TARGET_SR
        assert window_end - window_start <= 75 * TARGET_SR
        assert not wav[window_end - TARGET_SR // 100 : window_end + TARGET_SR // 100].any()  # in a pause


class _Redo:
    """A worker that answers each piece with the next of `answers`, noting the pieces."""

    def __init__(self, answers):
        self.answers, self.pieces = list(answers), []

    async def submit_many(self, pieces, _model_key):
        self.pieces += pieces
        return [self.answers.pop(0) for _piece in pieces]


def _request(worker):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(worker=worker, ready=True)))


@pytest.mark.asyncio
async def test_a_piece_that_stops_short_is_decoded_again_without_context():
    # Speech throughout; the cut at 10 s, each piece decoding 3 s of the other
    wav = np.arange(20 * TARGET_SR, dtype=np.float32)
    windows = [(0.0, 13.0), (7.0, 20.0)]
    prepared = _prepared(
        [(0.0, 10.0), (10.0, 20.0)], pieces=[wav[a:b] for a, b in _samples(windows)], windows_sec=windows
    )
    prepared.speech = _samples([(0.0, 20.0)])
    stopped = _result([" One", " two"], [0.5, 1.0])  # nothing after 1 s: 8.7 s of its speech unheard
    whole = _result([" four", " five"], [4.0, 12.0])  # to 19 s: 0.7 s unheard
    again = _result([" One", " two", " three"], [0.5, 1.0, 9.0])
    worker = _Redo([again])
    results = await routes._redo_stalled(_request(worker), [prepared], [stopped, whole], "parakeet-v3:fp32")
    assert results == [again, whole]
    assert [(piece[0], piece.size) for piece in worker.pieces] == [(0, 10 * TARGET_SR)]
    assert prepared.windows == _samples([(0.0, 10.0), (7.0, 20.0)])
    assert prepared.pieces[0].size == 10 * TARGET_SR
    text, _segments, _words = routes._stitch(prepared, results)
    assert text == "One two three four five"


@pytest.mark.asyncio
async def test_no_piece_is_decoded_again_when_the_rest_is_silence():
    windows = [(0.0, 13.0), (7.0, 20.0)]
    prepared = _prepared([(0.0, 10.0), (10.0, 20.0)], windows_sec=windows)
    prepared.speech = _samples([(0.0, 1.5), (10.5, 19.0)])  # quiet from 1.5 s to the cut
    results = [_result([" One", " two"], [0.5, 1.0]), _result([" four", " five"], [4.0, 11.5])]
    worker = _Redo([])
    assert await routes._redo_stalled(_request(worker), [prepared], results, "parakeet-v3:fp32") == results
    assert worker.pieces == [] and prepared.windows == _samples(windows)
