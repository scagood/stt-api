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


@pytest.mark.parametrize(
    "first_three, second_start",
    [
        # Lost: "three" (at 20.02 s) is at the cut, 20.00, on the first piece's
        # grid, so not before it; 15.03 + 4.96 = 19.99 on the second's, before it.
        (20.0, 15.03),
        # Duplicated: before the cut on the first piece's grid, 19.92; at
        # 15.07 + 4.96 = 20.03, after it, on the second's.
        (19.92, 15.07),
    ],
)
def test_a_word_at_a_cut_in_speech_is_kept_once(first_three, second_start):
    # A length cut at 20 s, in speech: each piece decodes 5 s of the other and
    # times each word on its own 80 ms grid, from its own window's start.
    first = _result([" one", " two", " three", " four", " five"], [18.4, 19.2, first_three, 20.8, 21.6])
    second = _result([" one", " two", " three", " four", " five"], [3.36, 4.16, 4.96, 5.76, 6.56])
    prepared = _prepared([(0.0, 20.0), (20.0, 40.0)], windows_sec=[(0.0, 25.0), (second_start, 40.0)])
    text, segments, words = routes._stitch(prepared, [first, second])
    assert text == "one two three four five"
    assert [s["segment"] for s in segments] == ["one two", "three four five"]
    assert [w["word"] for w in words] == ["one", "two", "three", "four", "five"]
    assert [round(w["start"], 2) for w in words][:2] == [18.4, 19.2]
    assert 20.0 <= words[2]["start"] <= 20.03 < words[2]["end"]


def test_a_word_said_again_at_a_cut_is_matched_with_itself_not_its_twin():
    # "I I I" across the cut: the first piece's first "I" (19.84) and the
    # second's middle one (15.03 + 5.04 = 20.07) are nearest the cut, but are
    # not the same "I"; taken as one, an "I" would be lost.
    first = _result([" it", " I", " I", " I"], [19.44, 19.84, 20.08, 20.4])
    second = _result([" it", " I", " I", " I"], [4.4, 4.8, 5.04, 5.44])
    prepared = _prepared([(0.0, 20.0), (20.0, 40.0)], windows_sec=[(0.0, 25.0), (15.03, 40.0)])
    text, _segments, words = routes._stitch(prepared, [first, second])
    assert text == "it I I I"
    assert _timed(words) == [("it", 19.44), ("I", 19.84), ("I", 20.07), ("I", 20.47)]


@pytest.mark.parametrize(
    "first_words, second_words, expected",
    [
        # Only the second piece hears X, after the cut: it keeps it
        (
            [(" A", 19.5), (" B", 20.4)],
            [(" A", 4.5), (" X", 5.1), (" B", 5.4)],
            [("A", 19.5), ("X", 20.1), ("B", 20.4)],
        ),
        # Only the first hears Y, before the cut: it keeps it, with a match after the cut or none
        (
            [(" A", 19.7), (" Y", 19.9), (" B", 20.4)],
            [(" A", 4.7), (" B", 5.4)],
            [("A", 19.7), ("Y", 19.9), ("B", 20.4)],
        ),
        ([(" A", 19.7), (" Y", 19.9)], [(" A", 4.7)], [("A", 19.7), ("Y", 19.9)]),
        # They hear it differently, after the cut: the second's, timed as it heard it
        (
            [(" gonna", 20.1), (" B", 20.5)],
            [(" going", 5.08), (" to", 5.2), (" B", 5.5)],
            [("going", 20.08), ("to", 20.2), ("B", 20.5)],
        ),
    ],
)
def test_words_only_one_piece_heard_near_a_cut_stay_in_the_range_they_start_in(
    first_words, second_words, expected
):
    first, second = (_result(*map(list, zip(*words))) for words in (first_words, second_words))
    prepared = _prepared([(0.0, 20.0), (20.0, 40.0)], windows_sec=[(0.0, 25.0), (15.0, 40.0)])
    text, _segments, words = routes._stitch(prepared, [first, second])
    assert text == " ".join(word for word, _start in expected)
    assert _timed(words) == expected


def test_a_word_at_each_cut_of_three_pieces_is_kept_once():
    # "two" would be lost at the cut at 20 s, and "four" kept twice at 40 s
    first = _result([" one", " two", " three"], [19.2, 20.0, 20.8])
    second = _result([" one", " two", " three", " four", " five"], [4.16, 4.96, 5.76, 24.88, 25.68])
    third = _result([" four", " five"], [4.96, 5.68])
    prepared = _prepared(
        [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)], windows_sec=[(0.0, 25.0), (15.03, 45.0), (35.07, 60.0)]
    )
    text, segments, _words = routes._stitch(prepared, [first, second, third])
    assert text == "one two three four five"
    assert [s["segment"] for s in segments] == ["one", "two three", "four five"]


def test_a_word_at_a_cut_is_kept_once_beside_a_piece_decoded_without_context():
    # The first piece was decoded again as just its range (_redo_stalled), so
    # hears up to the cut: "two" starts before it there, after it in the second
    first = _result([" one", " two"], [19.2, 19.92])
    second = _result([" one", " two", " three"], [4.16, 4.96, 5.76])
    prepared = _prepared([(0.0, 20.0), (20.0, 40.0)], windows_sec=[(0.0, 20.0), (15.07, 40.0)])
    text, segments, _words = routes._stitch(prepared, [first, second])
    assert text == "one two three"
    assert [s["segment"] for s in segments] == ["one", "two three"]


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
    quiet = TARGET_SR // 100
    for (start, end), (window_start, window_end) in zip(prepared.ranges[:-1], prepared.windows[:-1]):
        assert window_start <= start and window_end >= end + 5 * TARGET_SR
        assert window_end - window_start <= 75 * TARGET_SR
        assert not wav[window_end - quiet : window_end + quiet].any()  # in a pause
    for window_start, _window_end in prepared.windows[1:]:
        assert not wav[window_start - quiet : window_start + quiet].any()


class _Redo:
    """A worker that answers each piece with the next of `answers`, then with
    nothing heard, noting the pieces."""

    def __init__(self, answers):
        self.answers, self.pieces = list(answers), []

    async def submit_many(self, pieces, _model_key):
        self.pieces += pieces
        return [self.answers.pop(0) if self.answers else _result([], []) for _piece in pieces]


def _request(worker, pool=None):
    state = SimpleNamespace(worker=worker, ready=True, audio_pool=pool)  # None: asyncio's own pool
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _words(count, first, step=1.0):
    return _result([f" w{i}" for i in range(count)], [first + step * i for i in range(count)])


def _heard(origin, words):
    """A decode from `origin` s of `words`, each (token, its start in the audio)."""
    return _result([token for token, _at in words], [round(at - origin, 6) for _token, at in words])


def _decoded_spans(worker):
    """Where each piece of audio the worker was given starts and ends, in s
    (its samples are their own index: _two_pieces, _one_piece)."""
    return [(round(piece[0] / TARGET_SR, 2), round((piece[0] + piece.size) / TARGET_SR, 2)) for piece in worker.pieces]


def _two_pieces(pieces=None):
    """Cut at 10 s, each piece decoding 3 s of the other."""
    windows = [(0.0, 13.0), (7.0, 20.0)]
    wav = np.arange(20 * TARGET_SR, dtype=np.float32)  # each sample its own index, so a piece shows where it starts
    pieces = pieces or [wav[a:b] for a, b in _samples(windows)]
    prepared = _prepared([(0.0, 10.0), (10.0, 20.0)], pieces=pieces, windows_sec=windows)
    prepared.speech = _samples([(0.0, 20.0)])  # speech throughout
    return prepared


def _one_piece(seconds, speech=None):
    """A clip short enough to be one piece, as _prepare_audio leaves it: no VAD yet."""
    wav = np.arange(int(seconds * TARGET_SR), dtype=np.float32)
    prepared = _prepared([(0.0, seconds)], pieces=[wav])
    prepared.waveform, prepared.speech = wav, speech
    return prepared


# A piece that decoded context and skipped speech is first decoded again as
# just its range, without context, which is kept or not as a whole.


@pytest.mark.asyncio
async def test_a_piece_that_stops_short_is_decoded_again_without_context():
    prepared = _two_pieces()
    stopped = _result([" One", " two"], [0.5, 1.0])  # nothing after 1 s: 8.7 s of its speech unheard
    whole = _words(10, 3.0)  # a word a second, from 10 s
    again = _result([" One", " two", " three", " four", " five"], [0.5, 1.0, 3.5, 6.0, 8.5])
    worker = _Redo([again])
    results = await routes._redo_stalled(_request(worker), [prepared], [stopped, whole], "parakeet-v3:fp32")
    assert results == [again, whole]
    assert _decoded_spans(worker) == [(0.0, 10.0)]
    assert prepared.windows == _samples([(0.0, 10.0), (7.0, 20.0)])
    assert prepared.pieces[0].size == 10 * TARGET_SR
    text, _segments, _words_ = routes._stitch(prepared, results)
    assert text == "One two three four five " + " ".join(f"w{i}" for i in range(10))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "again",
    [
        _result([" One", " two"], [0.5, 1.0]),  # music, laughter or noise: no words to find
        _result([" One", " two", " and"], [0.5, 1.0, 9.6]),  # a word made up as the input ends (#68)
    ],
)
async def test_a_redo_that_hears_no_more_words_is_dropped(again):
    prepared = _two_pieces()
    stopped = _result([" One", " two"], [0.5, 1.0])
    results = [stopped, _words(10, 3.0)]
    worker = _Redo([again])  # then the stretch on its own: nothing there either
    assert await routes._redo_stalled(_request(worker), [prepared], results, "parakeet-v3:fp32") == results
    assert _decoded_spans(worker) == [(0.0, 10.0), (0.0, 12.0)]  # its range, then 1.32-10 s and 2 s either side
    assert prepared.windows == _samples([(0.0, 13.0), (7.0, 20.0)])
    assert prepared.pieces[0].size == 13 * TARGET_SR


@pytest.mark.asyncio
async def test_a_piece_that_skips_speech_mid_way_is_decoded_again():
    # Its last word is at its range's end, but 11.3-18 s of speech has none
    prepared = _two_pieces()
    skipping = _result([" a", " b", " c", " x", " y", " z"], [3.0, 3.5, 4.0, 11.0, 11.5, 12.5])
    again = _words(10, 0.0)
    worker = _Redo([again])
    results = await routes._redo_stalled(_request(worker), [prepared], [_words(10, 0.5), skipping], "parakeet-v3:fp32")
    assert results[1] is again and len(worker.pieces) == 1
    assert prepared.windows == _samples([(0.0, 13.0), (10.0, 20.0)])


@pytest.mark.asyncio
async def test_the_next_word_timed_a_frame_earlier_is_not_a_new_one():
    # The second piece skips 11.32-16.04 s. Decoded from 10 s, not 7 s, the
    # redo times "three" a frame earlier (16.00 s), just inside that stretch:
    # with a made-up "uh" it would make two new words. So does the stretch's
    # own redo, from 9.32 s.
    prepared = _two_pieces()
    heard = [3.0, 3.5, 4.0, 9.04, 9.5, 10.0, 11.0, 12.0]  # from 7 s: 10, 10.5, 11, 16.04 ...
    skipping = _result([f" w{i}" for i in range(3)] + [" three"] + [f" v{i}" for i in range(4)], heard)
    again = _result(
        [f" w{i}" for i in range(3)] + [" uh", " three"] + [f" v{i}" for i in range(4)],
        [0.0, 0.5, 1.0, 3.04, 6.0, 6.5, 7.0, 8.0, 9.0],
    )
    stretch = _heard(9.32, [(" w0", 10.0), (" w1", 10.5), (" w2", 11.0), (" uh", 13.04), (" three", 16.0), (" v0", 16.5)])
    results = [_words(10, 0.5), skipping]
    worker = _Redo([again, stretch])
    assert await routes._redo_stalled(_request(worker), [prepared], results, "parakeet-v3:fp32") == results
    assert _decoded_spans(worker) == [(10.0, 20.0), (9.32, 18.04)]
    assert prepared.windows == _samples([(0.0, 13.0), (7.0, 20.0)])


@pytest.mark.asyncio
async def test_a_redo_that_loses_more_words_than_it_finds_is_dropped():
    prepared = _two_pieces()
    stopped = _result([f" w{i}" for i in range(6)], [0.3, 0.7, 1.1, 1.5, 1.9, 2.3])  # nothing after 2.3 s
    again = _result([" x", " y"], [8.0, 8.5])  # two new words, but six lost
    results = [stopped, _words(10, 3.0)]
    worker = _Redo([again])
    assert await routes._redo_stalled(_request(worker), [prepared], results, "parakeet-v3:fp32") == results
    assert len(worker.pieces) == 2 and prepared.windows == _samples([(0.0, 13.0), (7.0, 20.0)])


@pytest.mark.asyncio
async def test_no_piece_is_decoded_again_when_the_rest_is_silence():
    prepared = _two_pieces()
    prepared.speech = _samples([(0.0, 1.5), (10.5, 19.0)])  # quiet from 1.5 s to the cut
    results = [_result([" One", " two"], [0.5, 1.0]), _words(9, 3.5)]
    worker = _Redo([])
    assert await routes._redo_stalled(_request(worker), [prepared], results, "parakeet-v3:fp32") == results
    assert worker.pieces == [] and prepared.windows == _samples([(0.0, 13.0), (7.0, 20.0)])


# Then each stretch still skipped, in any piece, is decoded again on its own,
# and the words heard there are put in among the piece's own.


@pytest.mark.asyncio
async def test_what_a_range_redo_skips_again_is_decoded_again_on_its_own_and_put_in():
    prepared = _two_pieces()
    stopped = _result([" One", " two"], [0.5, 1.0])  # nothing after 1 s
    whole = _words(10, 3.0)
    # Heard from 0 s, as the piece was: its first words go by the piece's own
    # decode. "and" is past the cut, where the next piece heard nothing: its
    # context, which _trimmed drops. "so", the redo's last word, may be one it
    # made up as its input ended (#68).
    stretch = _result([" one", " two", " three", " four", " and", " so"], [0.5, 1.0, 5.0, 9.0, 10.55, 11.5])
    worker = _Redo([_result([" One", " two"], [0.5, 1.0]), stretch])
    results = await routes._redo_stalled(_request(worker), [prepared], [stopped, whole], "parakeet-v3:fp32")
    assert _decoded_spans(worker) == [(0.0, 10.0), (0.0, 12.0)]
    assert results[1] is whole
    assert results[0].tokens == [" One", " two", " three", " four", " and"]
    assert results[0].timestamps == [0.5, 1.0, 5.0, 9.0, 10.55]
    assert prepared.windows == _samples([(0.0, 13.0), (7.0, 20.0)])  # the piece's own, as its times are
    text, _segments, _words_ = routes._stitch(prepared, results)
    assert text == "One two three four " + " ".join(f"w{i}" for i in range(10))


def _filler(name, start, stop, step=0.5):
    return [(f" {name}{i}", start + step * i) for i in range(int(round((stop - start) / step)) + 1)]


@pytest.mark.asyncio
@pytest.mark.parametrize("gap", [0.16, 0.24])
async def test_words_close_to_those_either_side_of_a_skip_are_put_in(gap):
    # #79 review: the piece hears up to "under|stand|ing" (its last token at
    # 19.84 s), skips, and hears again from "Rome" (28 s). The redo hears
    # "the" just after that last token, and "of" 0.24 s before "Rome", which
    # it times a frame early.
    before = _filler("a", 0.5, 18.5) + [(" we", 18.9), (" were", 19.2), (" under", 19.52), ("stand", 19.68), ("ing", 19.84)]
    after = [(" Rome", 28.0)] + _filler("b", 28.5, 39.5)
    middle = [(" the", 19.84 + gap), (" history", 20.4), (" of", 21.0)] + _filler("c", 21.5, 27.0) + [(" part", 27.4), (" of", 27.76)]
    prepared = _one_piece(40.0, speech=_samples([(0.0, 40.0)]))
    redo = _heard(18.16, [(" a36", 18.5)] + before[-5:] + middle + [(" Rome", 27.92), (" b0", 28.5), (" b1", 29.0)])
    worker = _Redo([redo])
    results = await routes._redo_stalled(_request(worker), [prepared], [_heard(0.0, before + after)], "parakeet-v3:fp32")
    assert _decoded_spans(worker) == [(18.16, 30.0)]  # 20.16-28 s and 2 s either side
    text, _segments, words = routes._stitch(prepared, results)
    assert "a36 we were understanding the history of c0" in text
    assert "c11 part of Rome b0 b1 b2" in text
    assert [w["word"] for w in words].count("Rome") == 1
    assert [w["start"] for w in words] == sorted(w["start"] for w in words)


@pytest.mark.asyncio
@pytest.mark.parametrize("standing", [19.76, 19.92, 20.0])
async def test_speech_the_piece_heard_as_another_word_is_not_put_in_again(standing):
    # The redo hears "understanding" as "under standing", and "Rome" as "roam"
    before = _filler("a", 0.5, 18.5) + [(" under", 19.52), ("stand", 19.68), ("ing", 19.84)]
    after = [(" Rome", 28.0)] + _filler("b", 28.5, 39.5)
    prepared = _one_piece(40.0, speech=_samples([(0.0, 40.0)]))
    redo = _heard(18.16, [(" under", 19.52), (" standing", standing), (" the", 20.4), (" past", 24.0), (" roam", 28.08), (" b0", 28.5)])
    worker = _Redo([redo])
    results = await routes._redo_stalled(_request(worker), [prepared], [_heard(0.0, before + after)], "parakeet-v3:fp32")
    words = routes._stitch(prepared, results)[0].split()
    assert words[35:40] == ["a35", "a36", "understanding", "the", "past"]
    assert words[40:42] == ["Rome", "b0"] and len(words) == 37 + 3 + 1 + 23


@pytest.mark.asyncio
@pytest.mark.parametrize("skips", ["the start of the next piece", "the end of this piece"])
async def test_a_word_straddling_a_cut_is_kept_once(skips):
    # #79 review: "X" starts at 10.07 s, just past the cut, and the pieces and
    # the redo time it either side of it.
    prepared = _two_pieces()
    first = [(f" a{i}", 0.5 + 0.5 * i) for i in range(19)]  # 0.5-9.5 s
    later = [(" Y", 10.6), (" Z", 11.5)] + [(f" b{i}", 14.0 + 0.5 * i) for i in range(12)]  # to 19.5 s
    if skips == "the start of the next piece":
        # The first piece times X past the cut (its context); the second hears
        # nothing from its start to 14 s, and its redo times X before the cut.
        left = _heard(0.0, first + [(" X", 10.08), (" Y", 10.6), (" Z", 11.5)])
        right = _heard(7.0, later[2:])
        stretch = _heard(8.0, first[-3:] + [(" X", 9.96)] + later[:4])
        answers, at = [_heard(10.0, later[2:]), stretch], 1  # its range alone hears nothing new
    else:
        # The first piece stops after 5 s and hears again from X, past the cut
        # (its context); its redo times X right at the cut, the second piece past it.
        left = _heard(0.0, first[:10] + [(" X", 10.06), (" Y", 10.6), (" Z", 11.5)])
        right = _heard(7.0, [(" X", 10.08)] + later)
        stretch = _heard(3.32, first[6:] + [(" X", 10.0), (" Y", 10.6), (" Z", 11.5)])
        answers, at = [_heard(0.0, first[:10]), stretch], 0
    worker = _Redo(answers)
    results = await routes._redo_stalled(_request(worker), [prepared], [left, right], "parakeet-v3:fp32")
    assert results[at] is not (left, right)[at]  # the stretch's words were put in
    words = routes._stitch(prepared, results)[0].split()
    assert words.count("X") == 1 and words.count("Y") == 1 and words.count("Z") == 1
    assert words == [w.strip() for w, _at in first + [(" X", 10.07)] + later]


@pytest.mark.asyncio
async def test_a_word_the_next_piece_heard_differently_past_the_cut_is_kept_once():
    # The first piece stops after 5 s. Its redo hears "X" as "ex", just before
    # the cut; the second piece heard "X" just past it, in its own range.
    prepared = _two_pieces()
    first = [(f" a{i}", 0.5 + 0.5 * i) for i in range(19)]  # 0.5-9.5 s
    later = [(" X", 10.06)] + [(f" b{i}", 10.5 + 0.5 * i) for i in range(19)]  # to 19.5 s
    stretch = _heard(3.32, first[6:] + [(" ex", 9.98), (" b0", 10.5), (" b1", 11.0)])
    worker = _Redo([_heard(0.0, first[:10]), stretch])
    results = [_heard(0.0, first[:10]), _heard(7.0, later)]
    results = await routes._redo_stalled(_request(worker), [prepared], results, "parakeet-v3:fp32")
    assert routes._stitch(prepared, results)[0].split() == [w.strip() for w, _at in first + later]


@pytest.mark.asyncio
@pytest.mark.parametrize("uh", [21.35, 21.92])
async def test_a_word_made_up_in_the_redo_s_margin_is_not_put_in(uh):
    # #79 review: a word every 0.7 s; the piece skips 10-20 s. Its redo hears
    # it all, and makes up "uh" in its margin, mid-way or as its input ends.
    words = [(f" w{i}", 0.7 * i) for i in range(86)]
    heard = [(word, at) for word, at in words if not 10.0 <= at < 20.0]
    prepared = _one_piece(60.0, speech=_samples([(0.0, 60.0)]))
    redo = _heard(8.12, sorted([(word, at) for word, at in words if 8.12 <= at < 22.3] + [(" uh", uh)], key=lambda w: w[1]))
    worker = _Redo([redo])
    results = await routes._redo_stalled(_request(worker), [prepared], [_heard(0.0, heard)], "parakeet-v3:fp32")
    assert _decoded_spans(worker) == [(8.12, 22.3)]
    assert routes._stitch(prepared, results)[0].split() == [word.strip() for word, _at in words]


@pytest.mark.asyncio
async def test_a_word_made_up_as_the_redo_meets_the_end_of_the_audio_is_not_put_in():
    # The piece stops hearing at 20 s of 30; its redo hears the rest, and "uh"
    # as its input ends, where the audio does (#68).
    words = [(f" w{i}", 0.5 + 0.5 * i) for i in range(59)]  # to 29.5 s
    prepared = _one_piece(30.0, speech=_samples([(0.0, 30.0)]))
    redo = _heard(18.32, [(word, at) for word, at in words if at >= 18.32] + [(" uh", 29.8)])
    worker = _Redo([redo])
    results = await routes._redo_stalled(_request(worker), [prepared], [_heard(0.0, words[:40])], "parakeet-v3:fp32")
    assert routes._stitch(prepared, results)[0].split() == [word.strip() for word, _at in words]


@pytest.mark.asyncio
@pytest.mark.parametrize("day", [9.24, 9.32])
async def test_a_word_the_redo_splits_in_two_is_not_put_in_again(day):
    # #79 review: the piece hears one-token "today" at 9 s, then skips to
    # 20 s. The redo hears it as "to day".
    before, after = _filler("a", 0.5, 8.5) + [(" today", 9.0)], _filler("b", 20.0, 29.5)
    prepared = _one_piece(30.0, speech=_samples([(0.0, 30.0)]))
    middle = _filler("c", 10.0, 19.5)
    redo = _heard(7.32, [(" a14", 7.5), (" a15", 8.0), (" a16", 8.5), (" to", 9.0), (" day", day)] + middle + after[:5])
    worker = _Redo([redo])
    results = await routes._redo_stalled(_request(worker), [prepared], [_heard(0.0, before + after)], "parakeet-v3:fp32")
    assert routes._stitch(prepared, results)[0].split() == [w.strip() for w, _at in before + middle + after]


@pytest.mark.asyncio
async def test_words_the_redo_times_early_as_its_input_ends_are_not_put_in_again():
    # #79 review: the piece skips 20-28 s and hears again from "Rome fell.
    # Then Caesar"; the redo times "Then" and "Caesar" 0.48 s early.
    before = _filler("a", 0.5, 19.5)
    after = [(" Rome", 28.0), (" fell.", 28.4), (" Then", 28.9), (" Caesar", 29.3)] + _filler("b", 29.8, 39.8)
    prepared = _one_piece(40.0, speech=_samples([(0.0, 40.0)]))
    middle = _filler("c", 20.3, 27.3)
    redo = _heard(17.82, [(" a35", 18.0), (" a36", 18.5)] + middle + after[:2] + [(" Then", 28.42), (" Caesar", 28.82), (" b0", 29.32)])
    worker = _Redo([redo])
    results = await routes._redo_stalled(_request(worker), [prepared], [_heard(0.0, before + after)], "parakeet-v3:fp32")
    assert routes._stitch(prepared, results)[0].split() == [w.strip() for w, _at in before + middle + after]


@pytest.mark.asyncio
@pytest.mark.parametrize(("heard_as", "at"), [(" roam", 25.04), (" Rome", 25.12)])
async def test_a_word_timed_early_as_the_redo_meets_the_end_of_the_audio_is_not_put_in_again(heard_as, at):
    # The piece skips 17-25.6 s of a 26 s clip and hears "Rome" last. Its
    # redo, ending where the audio does, hears it about half a second early,
    # as another word or the same, then makes up "uh".
    before = _filler("a", 0.5, 17.0)
    middle = _filler("c", 17.5, 24.5)
    prepared = _one_piece(26.0, speech=_samples([(0.0, 26.0)]))
    redo = _heard(15.28, before[-3:] + middle + [(heard_as, at), (" uh", 25.8)])
    worker = _Redo([redo])
    results = await routes._redo_stalled(_request(worker), [prepared], [_heard(0.0, before + [(" Rome", 25.6)])], "parakeet-v3:fp32")
    assert routes._stitch(prepared, results)[0].split() == [w.strip() for w, _at in before + middle] + ["Rome"]


@pytest.mark.asyncio
async def test_a_word_timed_early_from_the_redo_s_last_seconds_is_not_put_in_again():
    # The piece skips 17-25 s of a 26 s clip and hears "Rome" last, at
    # 24.96 s. Its redo hears it as "roam" 0.56 s early: in its last 1.5 s
    # as said, but before them as timed.
    before = _filler("a", 0.5, 17.0)
    middle = _filler("c", 17.5, 24.0)
    prepared = _one_piece(26.0, speech=_samples([(0.0, 26.0)]))
    redo = _heard(15.28, before[-3:] + middle + [(" roam", 24.4), (" uh", 25.8)])
    worker = _Redo([redo])
    results = await routes._redo_stalled(_request(worker), [prepared], [_heard(0.0, before + [(" Rome", 24.96)])], "parakeet-v3:fp32")
    assert routes._stitch(prepared, results)[0].split() == [w.strip() for w, _at in before + middle] + ["Rome"]


@pytest.mark.asyncio
async def test_the_second_half_of_a_word_the_redo_splits_is_not_put_in_either():
    # The piece hears one-token "today" at 9 s, then skips to 20 s; the redo
    # hears "to" at 9.16 s and "day" 0.4 s after it.
    before, after = _filler("a", 0.5, 8.5) + [(" today", 9.0)], _filler("b", 20.0, 29.5)
    prepared = _one_piece(30.0, speech=_samples([(0.0, 30.0)]))
    middle = _filler("c", 10.0, 19.5)
    redo = _heard(7.32, [(" a14", 7.5), (" a15", 8.0), (" a16", 8.5), (" to", 9.16), (" day", 9.56)] + middle + after[:5])
    worker = _Redo([redo])
    results = await routes._redo_stalled(_request(worker), [prepared], [_heard(0.0, before + after)], "parakeet-v3:fp32")
    assert routes._stitch(prepared, results)[0].split() == [w.strip() for w, _at in before + middle + after]


@pytest.mark.asyncio
async def test_the_second_half_of_a_split_word_after_another_heard_differently_is_not_put_in():
    # As above, with "the" before "today" heard by the redo as "thee".
    before, after = _filler("a", 0.5, 8.0) + [(" the", 8.5), (" today", 9.0)], _filler("b", 20.0, 29.5)
    prepared = _one_piece(30.0, speech=_samples([(0.0, 30.0)]))
    middle = _filler("c", 10.0, 19.5)
    redo = _heard(7.32, [(" a14", 7.5), (" a15", 8.0), (" thee", 8.5), (" to", 9.16), (" day", 9.56)] + middle + after[:5])
    worker = _Redo([redo])
    results = await routes._redo_stalled(_request(worker), [prepared], [_heard(0.0, before + after)], "parakeet-v3:fp32")
    assert routes._stitch(prepared, results)[0].split() == [w.strip() for w, _at in before + middle + after]


@pytest.mark.asyncio
async def test_a_word_the_redo_splits_and_times_sooner_is_not_put_in_again():
    # As above, with "to" a few frames before the piece's "today".
    before, after = _filler("a", 0.5, 8.5) + [(" today", 9.0)], _filler("b", 20.0, 29.5)
    prepared = _one_piece(30.0, speech=_samples([(0.0, 30.0)]))
    middle = _filler("c", 10.0, 19.5)
    redo = _heard(7.32, [(" a14", 7.5), (" a15", 8.0), (" a16", 8.5), (" to", 8.76), (" day", 9.16)] + middle + after[:5])
    worker = _Redo([redo])
    results = await routes._redo_stalled(_request(worker), [prepared], [_heard(0.0, before + after)], "parakeet-v3:fp32")
    assert routes._stitch(prepared, results)[0].split() == [w.strip() for w, _at in before + middle + after]


@pytest.mark.asyncio
async def test_a_word_split_across_a_cut_is_not_put_in_again():
    # The first piece stops after 5 s; its redo hears "word" just before the
    # cut. The second piece heard it as "wo" (its context) and "rd" (its own).
    prepared = _two_pieces()
    first = [(f" a{i}", 0.5 + 0.5 * i) for i in range(10)]  # 0.5-5 s
    middle = [(f" c{i}", 5.5 + 0.5 * i) for i in range(8)]  # 5.5-9 s
    later = [(" b0", 10.6), (" b1", 11.2)] + [(f" d{i}", 12.0 + 0.5 * i) for i in range(16)]
    left, right = _heard(0.0, first), _heard(7.0, middle[3:] + [(" wo", 9.88), (" rd", 10.12)] + later)
    stretch = _heard(3.32, first[6:] + middle + [(" word", 9.86)] + later[:3])
    worker = _Redo([_heard(0.0, first), stretch])
    results = await routes._redo_stalled(_request(worker), [prepared], [left, right], "parakeet-v3:fp32")
    words = routes._stitch(prepared, results)[0].split()
    assert words == [w.strip() for w, _at in first + middle] + ["rd"] + [w.strip() for w, _at in later]


@pytest.mark.asyncio
async def test_a_word_the_two_pieces_redos_hear_differently_at_their_cut_is_kept_once():
    # #79 review: both pieces skip 6-14 s, whole and on their own. The first's
    # redo hears "Holmes" just before the cut; the second's "homes" at it.
    prepared = _two_pieces()
    first, middle = _filler("a", 0.5, 5.5), _filler("c", 6.0, 9.5)
    later = [(" e0", 10.5), (" e1", 11.0), (" e2", 11.5), (" e3", 12.0)] + _filler("b", 14.0, 19.5)
    left, right = _heard(0.0, first), _heard(7.0, later[4:])
    answers = [
        _heard(0.0, first), _heard(10.0, later[4:]),  # their ranges alone: nothing new
        _heard(3.82, first[-4:] + middle + [(" Holmes", 9.84)] + later[:3]),
        _heard(8.0, middle[4:] + [(" homes", 10.0)] + later[:7]),
    ]
    results = await routes._redo_stalled(_request(_Redo(answers)), [prepared], [left, right], "parakeet-v3:fp32")
    words = routes._stitch(prepared, results)[0].split()
    assert words == [w.strip() for w, _at in first + middle] + ["Holmes"] + [w.strip() for w, _at in later]


@pytest.mark.asyncio
async def test_a_word_said_twice_across_a_cut_is_kept_twice():
    # #79 review: "to" at 9.9 s and again at 10.16 s, the cut between them. The
    # second piece hears nothing to 14 s; its redo hears both.
    prepared = _two_pieces()
    first, later = _filler("a", 0.5, 9.5) + [(" to", 9.9)], [(" c0", 10.6), (" c1", 11.2)] + _filler("b", 14.0, 19.5)
    left, right = _heard(0.0, first), _heard(7.0, later[2:])
    stretch = _heard(8.0, first[-4:] + [(" to", 10.16)] + later[:6])
    worker = _Redo([_heard(10.0, later[2:]), stretch])
    results = await routes._redo_stalled(_request(worker), [prepared], [left, right], "parakeet-v3:fp32")
    words = routes._stitch(prepared, results)[0].split()
    assert words == [w.strip() for w, _at in first] + ["to"] + [w.strip() for w, _at in later]


@pytest.mark.asyncio
async def test_a_word_the_neighbour_keeps_timed_early_at_the_cut_is_not_put_in_again():
    # The first piece times "X" 0.6 s early, before the cut, where it keeps
    # it. The second hears nothing to 14 s; its redo times X right, past the
    # cut: too far apart for _seam to see one word.
    prepared = _two_pieces()
    first, later = _filler("a", 0.5, 9.0) + [(" X", 9.44)], [(" c0", 10.6), (" c1", 11.2)] + _filler("b", 14.0, 19.5)
    left, right = _heard(0.0, first), _heard(7.0, later[2:])
    stretch = _heard(8.0, first[-4:-1] + [(" X", 10.04)] + later[:6])
    worker = _Redo([_heard(10.0, later[2:]), stretch])
    results = await routes._redo_stalled(_request(worker), [prepared], [left, right], "parakeet-v3:fp32")
    words = routes._stitch(prepared, results)[0].split()
    assert words == [w.strip() for w, _at in first + later]


@pytest.mark.asyncio
async def test_words_after_a_word_whose_full_stop_is_timed_late_are_put_in():
    # #79 review: the piece hears "Rome" and its "." a second later, then
    # skips to 30 s. The redo hears "The next thing" before that ".".
    before, after = _filler("a", 0.5, 19.5) + [(" Rome", 20.0), (".", 21.0)], _filler("b", 30.0, 39.5)
    prepared = _one_piece(40.0, speech=_samples([(0.0, 40.0)]))
    middle = [(" The", 20.6), (" next", 20.9), (" thing", 21.2)] + _filler("c", 21.7, 29.2)
    redo = _heard(18.32, [(" a36", 18.5), (" a37", 19.0), (" a38", 19.5), (" Rome", 20.0), (".", 20.16)] + middle + after[:3])
    worker = _Redo([redo])
    results = await routes._redo_stalled(_request(worker), [prepared], [_heard(0.0, before + after)], "parakeet-v3:fp32")
    words = routes._stitch(prepared, results)[0].split()
    assert words == [w.strip() for w, _at in before[:-2]] + ["Rome."] + [w.strip() for w, _at in middle + after]


@pytest.mark.asyncio
async def test_the_stretches_a_piece_skipped_count_together(caplog):
    # One word in each of two skipped stretches: two new words in all
    heard = _filler("a", 0.5, 19.5) + _filler("b", 24.0, 29.5) + _filler("c", 34.0, 39.5)
    prepared = _one_piece(40.0, speech=_samples([(0.0, 40.0)]))
    worker = _Redo([_heard(17.82, [(" Yes", 22.0), (" b0", 24.0)]), _heard(27.82, [(" Right", 32.0), (" c0", 34.0)])])
    results = await routes._redo_stalled(_request(worker), [prepared], [_heard(0.0, heard)], "parakeet-v3:fp32")
    assert _decoded_spans(worker) == [(17.82, 26.0), (27.82, 36.0)]
    text = routes._stitch(prepared, results)[0]
    assert "a38 Yes b0" in text and "b11 Right c0" in text
    assert "19.8-24.0 s, 29.8-34.0 s: 2 words" in caplog.text


@pytest.mark.asyncio
async def test_stretches_whose_redos_overlap_are_decoded_together():
    heard = _filler("a", 0.5, 9.5) + _filler("b", 14.0, 15.5) + _filler("c", 19.0, 29.5)
    prepared = _one_piece(30.0, speech=_samples([(0.0, 30.0)]))
    worker = _Redo([_heard(7.82, [(" x", 11.0), (" y", 17.5), (" c0", 19.0)])])
    results = await routes._redo_stalled(_request(worker), [prepared], [_heard(0.0, heard)], "parakeet-v3:fp32")
    assert _decoded_spans(worker) == [(7.82, 21.0)]  # 9.82-14 s and 15.82-19 s, with 2 s either side
    text = routes._stitch(prepared, results)[0]
    assert "a18 x b0" in text and "b3 y c0" in text


@pytest.mark.asyncio
async def test_a_one_piece_clip_that_skips_its_opening_has_it_decoded_again(monkeypatch):
    # #77: narration over a music bed, which a clip of 70 s heard only from
    # "Part one" at 10.4 s, and the 0-10.4 s alone in full.
    vad = []
    monkeypatch.setattr(routes, "speech_segments", lambda wav: vad.append(wav.size) or [(0, wav.size)])
    prepared = _one_piece(70.0)
    rest = [f" w{i}" for i in range(58)]
    heard = _result([" Part", " one", "."] + rest, [10.4, 10.8, 11.0] + [12.0 + i for i in range(58)])
    opening = [" Nine", "teen", " eighty", "-", "four", ",", " by", " George", " Or", "well", ".",
               " Read", " by", " Stephen", " Fry", "."]
    again = _result(
        opening + [" Part", " one", "."],  # "Part" a frame earlier, at 10.32 s
        [0.5, 0.8, 1.2, 1.5, 1.7, 1.9, 2.4, 2.8, 3.3, 3.5, 3.8, 5.0, 5.4, 5.8, 6.6, 6.9, 10.32, 10.8, 11.0],
    )
    worker = _Redo([again])
    results = await routes._redo_stalled(_request(worker), [prepared], [heard], "parakeet-v3:int8")
    assert vad == [70 * TARGET_SR]  # on the clip, once it had 3 s or more with no token
    assert _decoded_spans(worker) == [(0.0, 12.4)]  # 0-10.4 s and 2 s after
    text, _segments, words = routes._stitch(prepared, results)
    assert text.startswith("Nineteen eighty-four, by George Orwell. Read by Stephen Fry. Part one. w0 w1")
    assert [(w["word"], w["start"]) for w in words[:2]] == [("Nineteen", 0.5), ("eighty-four,", 1.2)]
    assert sum(w["word"] == "Part" for w in words) == 1


class _NoPool:
    def submit(self, *_args, **_kwargs):
        pytest.fail("waited on the audio pool")


@pytest.mark.asyncio
async def test_a_one_piece_clip_with_no_long_stretch_without_words_is_left_as_it_is(monkeypatch):
    # No VAD, and no wait on the audio pool behind other requests' audio
    monkeypatch.setattr(routes, "speech_segments", lambda _wav: pytest.fail("VAD ran"))
    results = [_words(30, 0.5)]  # a word a second throughout
    worker = _Redo([])
    assert await routes._redo_stalled(_request(worker, _NoPool()), [_one_piece(30.0)], results, "parakeet-v3:fp32") == results
    assert worker.pieces == []


@pytest.mark.asyncio
async def test_a_one_piece_clip_heard_as_nothing_is_not_decoded_again_as_it_was():
    # Music, say: the stretch is the whole clip, which would come back the same
    results = [_result([], [])]
    worker = _Redo([])
    prepared = _one_piece(30.0, speech=_samples([(0.0, 30.0)]))
    assert await routes._redo_stalled(_request(worker), [prepared], results, "parakeet-v3:fp32") == results
    assert worker.pieces == []


@pytest.mark.asyncio
async def test_a_stretch_by_the_window_s_edges_is_decoded_again_without_the_margin():
    # 0-8.68 s of a 10 s clip and 2 s either side is all of it: just the stretch, then
    results = [_result([" end"], [9.0])]
    worker = _Redo([_result([" a", " b", " c"], [1.0, 4.0, 8.9])])
    prepared = _one_piece(10.0, speech=_samples([(0.0, 10.0)]))
    results = await routes._redo_stalled(_request(worker), [prepared], results, "parakeet-v3:fp32")
    assert _decoded_spans(worker) == [(0.0, 9.0)]
    assert results[0].text == "a b end"  # "c" is the redo's "end", a frame early


@pytest.mark.asyncio
async def test_a_redo_never_costs_the_piece_its_own_words():
    stopped = _result([f" w{i}" for i in range(6)], [0.3, 0.7, 1.1, 1.5, 1.9, 2.3])  # nothing after 2.3 s
    again = _heard(0.62, [(" w4", 1.92), (" x", 8.0), (" y", 8.5), (" z", 9.4)])  # four of its own lost; "z" last
    worker = _Redo([again])
    prepared = _one_piece(10.0, speech=_samples([(0.0, 10.0)]))
    results = await routes._redo_stalled(_request(worker), [prepared], [stopped], "parakeet-v3:fp32")
    assert results[0].text == "w0 w1 w2 w3 w4 w5 x y"
    assert results[0].timestamps == pytest.approx([0.3, 0.7, 1.1, 1.5, 1.9, 2.3, 8.0, 8.5])


@pytest.mark.asyncio
@pytest.mark.parametrize("pieces", [1, 2])
async def test_a_redo_that_fails_keeps_what_was_heard(pieces, caplog):
    class Failing:
        async def submit_many(self, _pieces, _model_key):
            raise routes.ModelLoadError("parakeet-v3:fp32 could not be loaded")

    prepared = _two_pieces() if pieces == 2 else _one_piece(30.0, speech=_samples([(0.0, 30.0)]))
    results = [_result([" One", " two"], [0.5, 1.0]), _words(10, 3.0)][:pieces]
    assert await routes._redo_stalled(_request(Failing()), [prepared], results, "parakeet-v3:fp32") == results
    assert "keeping what" in caplog.text
    assert prepared.windows == (_samples([(0.0, 13.0), (7.0, 20.0)]) if pieces == 2 else _samples([(0.0, 30.0)]))


@pytest.mark.asyncio
async def test_a_redo_whose_splice_fails_keeps_what_was_heard(monkeypatch, caplog):
    def broken(*_args, **_kwargs):
        raise IndexError("a bug in putting words in")

    monkeypatch.setattr(routes, "_merged", broken)
    prepared = _one_piece(30.0, speech=_samples([(0.0, 30.0)]))
    results = [_result([" One", " two"], [0.5, 1.0])]
    worker = _Redo([_result([" One", " two", " three", " four"], [0.5, 1.0, 5.0, 9.0])])
    assert await routes._redo_stalled(_request(worker), [prepared], results, "parakeet-v3:fp32") == results
    assert "failed; keeping what was heard" in caplog.text


@pytest.mark.asyncio
async def test_a_batch_of_one_piece_clips_with_nothing_untimed_is_left_as_it_is(monkeypatch):
    monkeypatch.setattr(routes, "speech_segments", lambda _wav: pytest.fail("VAD ran"))
    files = [_one_piece(30.0) for _file in range(3)] + [_prepared([(0.0, 1.0)], pieces=[])]  # and a silent file
    files[-1].ranges = files[-1].windows = []
    results = [_words(30, 0.5) for _file in range(3)]
    assert await routes._redo_stalled(_request(_Redo([]), _NoPool()), files, results, "parakeet-v3:fp32") == results


@pytest.mark.asyncio
async def test_the_log_names_ten_pieces_and_counts_the_rest(caplog):
    files = [_one_piece(10.0, speech=_samples([(0.0, 10.0)])) for _file in range(12)]
    results = [_result([" end"], [9.0]) for _file in files]
    worker = _Redo([_result([" a", " b", " end"], [1.0, 4.0, 8.92]) for _file in files])
    results = await routes._redo_stalled(_request(worker), files, results, "parakeet-v3:fp32")
    assert all(result.text == "a b end" for result in results)
    assert caplog.text.count(": 2 words") == 10 and "; 2 more" in caplog.text


@pytest.mark.asyncio
async def test_whisper_is_never_decoded_again(monkeypatch):
    # No token times to find a skip by: a whole text is no skip
    monkeypatch.setattr(routes, "_stalled", lambda *_args: pytest.fail("Whisper was scanned"))
    whisper = next(name for name, config in routes.MODEL_CONFIGS.items() if config["family"] == "whisper")
    results = [_result([], [])]
    worker = _Redo([])
    prepared = _one_piece(30.0, speech=_samples([(0.0, 30.0)]))
    assert await routes._redo_stalled(_request(worker), [prepared], results, f"{whisper}:fp32") == results


@pytest.mark.asyncio
async def test_a_lone_full_stop_does_not_split_a_skipped_stretch():
    # The second piece skips 12.6-17 s of speech, but for a "." at 14.4 s: no
    # word starts there, so the stretch is one, not two under _STALL_SEC.
    prepared = _two_pieces()
    prepared.speech = _samples([(0.0, 20.0)])
    skipping = _result(
        [" a", " b", " c", " could", ".", " This", " y"],  # from 7 s: 10, 10.6, 11.2, 12.6, 14.4, 17, 18
        [3.0, 3.6, 4.2, 5.6, 7.4, 10.0, 11.0],
    )
    assert routes._stalled(prepared, [_words(10, 0.5), skipping]) == {1: _samples([(12.6 + 0.32, 17.0)])}
    again = _result(
        [" a", " b", " c", " could", ".", " The", " next", " thing", " This", " y"],
        [0.0, 0.6, 1.2, 2.6, 4.4, 4.5, 5.0, 5.5, 7.0, 8.0],
    )
    worker = _Redo([again])
    results = await routes._redo_stalled(_request(worker), [prepared], [_words(10, 0.5), skipping], "parakeet-v3:fp32")
    assert results[1] is again and len(worker.pieces) == 1


@pytest.mark.parametrize("mark", ["▁.", "▁-", "▁,", '▁"'])
def test_a_lone_marked_punctuation_token_does_not_split_a_skipped_stretch(mark):
    # As above, but the token opens a word of its own: it still has no word
    # character, so hears no speech.
    prepared = _two_pieces()
    prepared.speech = _samples([(0.0, 20.0)])
    skipping = _result(
        [" a", " b", " c", " could", mark, " This", " y"],  # from 7 s: 10, 10.6, 11.2, 12.6, 14.4, 17, 18
        [3.0, 3.6, 4.2, 5.6, 7.4, 10.0, 11.0],
    )
    assert routes._stalled(prepared, [_words(10, 0.5), skipping]) == {1: _samples([(12.6 + 0.32, 17.0)])}


def test_a_long_word_of_many_tokens_is_not_a_skipped_stretch():
    # A phone number read digit by digit, 11-16 s, is one word of eleven
    # tokens: each is speech heard, so nothing from its first to the next word
    # is skipped.
    prepared = _two_pieces()
    prepared.speech = _samples([(0.0, 20.0)])
    digits = list("18005550199")
    number = _result(
        [" at", " " + digits[0], *digits[1:], " call", " us", " today", "."],
        [3.5, 4.0, *[4.0 + 0.5 * i for i in range(1, 11)], 9.5, 10.0, 10.5, 11.0],
    )
    assert routes._stalled(prepared, [_words(10, 0.5), number]) == {}
