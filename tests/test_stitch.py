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
    """A worker that answers each piece with the next of `answers`, noting the pieces."""

    def __init__(self, answers):
        self.answers, self.pieces = list(answers), []

    async def submit_many(self, pieces, _model_key):
        self.pieces += pieces
        return [self.answers.pop(0) for _piece in pieces]


def _request(worker):
    state = SimpleNamespace(worker=worker, ready=True, audio_pool=None)  # None: asyncio's own pool
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _words(count, first, step=1.0):
    return _result([f" w{i}" for i in range(count)], [first + step * i for i in range(count)])


def _two_pieces(pieces=None):
    """Cut at 10 s, each piece decoding 3 s of the other."""
    windows = [(0.0, 13.0), (7.0, 20.0)]
    return _prepared([(0.0, 10.0), (10.0, 20.0)], pieces=pieces, windows_sec=windows)


@pytest.mark.asyncio
async def test_what_a_piece_skipped_is_decoded_again_on_its_own_and_put_in():
    wav = np.arange(20 * TARGET_SR, dtype=np.float32)
    prepared = _two_pieces([wav[a:b] for a, b in _samples([(0.0, 13.0), (7.0, 20.0)])])
    prepared.speech = _samples([(0.0, 20.0)])  # speech throughout
    stopped = _result([" One", " two"], [0.5, 1.0])  # nothing after 1 s: 8.7 s of its speech unheard
    whole = _words(10, 3.0)  # a word a second, from 10 s
    # Heard from 0 s, as the piece was: its first words go by the piece's own
    # decode, and "and" is past the cut, the next piece's.
    again = _result([" one", " two", " three", " four", " and"], [0.5, 1.0, 5.0, 9.0, 11.5])
    worker = _Redo([again])
    results = await routes._redo_stalled(_request(worker), [prepared], [stopped, whole], "parakeet-v3:fp32")
    # 1.32-10 s, and 2 s either side within the piece's window
    assert [(piece[0], piece.size) for piece in worker.pieces] == [(0, 12 * TARGET_SR)]
    assert results[1] is whole
    assert results[0].tokens == [" One", " two", " three", " four"]
    assert results[0].timestamps == [0.5, 1.0, 5.0, 9.0]
    assert prepared.windows == _samples([(0.0, 13.0), (7.0, 20.0)])  # the piece's own, as its times are
    assert prepared.pieces[0].size == 13 * TARGET_SR
    text, _segments, _words_ = routes._stitch(prepared, results)
    assert text == "One two three four " + " ".join(f"w{i}" for i in range(10))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "again",
    [
        _result([" One", " two"], [0.5, 1.0]),  # music, laughter or noise: no words to find
        _result([" One", " two", " and"], [0.5, 1.0, 9.6]),  # one word, as Parakeet makes up (#68)
    ],
)
async def test_a_redo_that_hears_no_more_words_is_dropped(again):
    wav = np.arange(20 * TARGET_SR, dtype=np.float32)
    prepared = _two_pieces([wav[a:b] for a, b in _samples([(0.0, 13.0), (7.0, 20.0)])])
    prepared.speech = _samples([(0.0, 20.0)])  # VAD called it all speech
    stopped = _result([" One", " two"], [0.5, 1.0])
    results = [stopped, _words(10, 3.0)]
    worker = _Redo([again])
    assert await routes._redo_stalled(_request(worker), [prepared], results, "parakeet-v3:fp32") == results
    assert len(worker.pieces) == 1  # decoded again, and the first decode kept
    assert prepared.windows == _samples([(0.0, 13.0), (7.0, 20.0)])
    assert prepared.pieces[0].size == 13 * TARGET_SR


@pytest.mark.asyncio
async def test_a_piece_that_skips_speech_mid_way_has_that_stretch_decoded_again():
    # Its last word is at its range's end, but 11.32-18 s of speech has none
    prepared = _two_pieces()
    prepared.speech = _samples([(0.0, 20.0)])
    skipping = _result([" a", " b", " c", " x", " y", " z"], [3.0, 3.5, 4.0, 11.0, 11.5, 12.5])
    # From 9.32 s: "b" and "c" the piece heard, "d" and "e" it skipped, and "x"
    # a frame earlier than it heard it.
    again = _result([" b", " c", " d", " e", " x"], [1.18, 1.68, 3.68, 5.68, 8.6])
    worker = _Redo([again])
    results = await routes._redo_stalled(_request(worker), [prepared], [_words(10, 0.5), skipping], "parakeet-v3:fp32")
    assert [piece.size for piece in worker.pieces] == [int(10.68 * TARGET_SR)]  # 9.32-20 s
    assert results[1].tokens == [" a", " b", " c", " d", " e", " x", " y", " z"]
    assert results[1].timestamps == pytest.approx([3.0, 3.5, 4.0, 6.0, 8.0, 11.0, 11.5, 12.5])
    assert results[1].text == "a b c d e x y z"
    assert prepared.windows == _samples([(0.0, 13.0), (7.0, 20.0)])


@pytest.mark.asyncio
async def test_the_next_word_timed_a_frame_earlier_is_not_a_new_one():
    # The second piece skips 11.32-16.04 s. Decoded again from 9.32 s, the
    # redo times "three" a frame earlier (16.00 s), just inside that stretch:
    # with a made-up "uh" it would make two new words.
    prepared = _two_pieces()
    prepared.speech = _samples([(0.0, 20.0)])
    heard = [3.0, 3.5, 4.0, 9.04, 9.5, 10.0, 11.0, 12.0]  # from 7 s: 10, 10.5, 11, 16.04 ...
    skipping = _result([f" w{i}" for i in range(3)] + [" three"] + [f" v{i}" for i in range(4)], heard)
    again = _result(
        [f" w{i}" for i in range(3)] + [" uh", " three"] + [f" v{i}" for i in range(2)],
        [0.68, 1.18, 1.68, 3.72, 6.68, 7.18, 7.68],
    )
    results = [_words(10, 0.5), skipping]
    worker = _Redo([again])
    assert await routes._redo_stalled(_request(worker), [prepared], results, "parakeet-v3:fp32") == results
    assert [piece.size for piece in worker.pieces] == [int(8.72 * TARGET_SR)]  # 9.32-18.04 s


@pytest.mark.asyncio
async def test_a_redo_never_costs_the_piece_its_own_words():
    prepared = _two_pieces()
    prepared.speech = _samples([(0.0, 20.0)])
    stopped = _result([f" w{i}" for i in range(6)], [0.3, 0.7, 1.1, 1.5, 1.9, 2.3])  # nothing after 2.3 s
    again = _result([" w4", " x", " y"], [1.3, 7.38, 7.88])  # from 0.62 s: two new words, four of its own lost
    worker = _Redo([again])
    results = await routes._redo_stalled(_request(worker), [prepared], [stopped, _words(10, 3.0)], "parakeet-v3:fp32")
    assert results[0].text == "w0 w1 w2 w3 w4 w5 x y"
    assert results[0].timestamps == pytest.approx([0.3, 0.7, 1.1, 1.5, 1.9, 2.3, 8.0, 8.5])


def _one_piece(seconds, speech=None):
    """A clip short enough to be one piece, as _prepare_audio leaves it: no VAD yet."""
    wav = np.full(int(seconds * TARGET_SR), 0.1, dtype=np.float32)
    prepared = _prepared([(0.0, seconds)], pieces=[wav])
    prepared.waveform, prepared.speech = wav, speech
    return prepared


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
    assert [piece.size for piece in worker.pieces] == [int(12.4 * TARGET_SR)]  # 0-10.4 s and 2 s after
    text, _segments, words = routes._stitch(prepared, results)
    assert text.startswith("Nineteen eighty-four, by George Orwell. Read by Stephen Fry. Part one. w0 w1")
    assert [(w["word"], w["start"]) for w in words[:2]] == [("Nineteen", 0.5), ("eighty-four,", 1.2)]
    assert sum(w["word"] == "Part" for w in words) == 1


@pytest.mark.asyncio
async def test_vad_runs_on_a_one_piece_clip_only_once_it_has_a_long_stretch_without_words(monkeypatch):
    monkeypatch.setattr(routes, "speech_segments", lambda _wav: pytest.fail("VAD ran"))
    results = [_words(30, 0.5)]  # a word a second throughout
    worker = _Redo([])
    assert await routes._redo_stalled(_request(worker), [_one_piece(30.0)], results, "parakeet-v3:fp32") == results
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
    assert [piece.size for piece in worker.pieces] == [9 * TARGET_SR]
    assert results[0].text == "a b end"  # "c" is the redo's "end", a frame early


@pytest.mark.asyncio
async def test_each_stretch_is_decoded_again_and_judged_on_its_own(caplog):
    # 0-5 s and 8.32-20 s with no word; again, the first hears two, the second one
    heard = _result([f" w{i}" for i in range(24)], [5.0, 6.0, 7.0, 8.0] + [20.0 + i for i in range(20)])
    worker = _Redo([_result([" a", " b"], [1.0, 2.0]), _result([" w3", " uh"], [1.68, 5.0])])
    prepared = _one_piece(40.0, speech=_samples([(0.0, 40.0)]))
    results = await routes._redo_stalled(_request(worker), [prepared], [heard], "parakeet-v3:fp32")
    assert [piece.size for piece in worker.pieces] == [7 * TARGET_SR, int(15.68 * TARGET_SR)]
    assert results[0].text == "a b " + " ".join(f"w{i}" for i in range(24))
    assert "0.0-5.0 s: 2 words, 8.3-20.0 s: 1 words" in caplog.text


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
async def test_no_piece_is_decoded_again_when_the_rest_is_silence():
    prepared = _two_pieces()
    prepared.speech = _samples([(0.0, 1.5), (10.5, 19.0)])  # quiet from 1.5 s to the cut
    results = [_result([" One", " two"], [0.5, 1.0]), _words(9, 3.5)]
    worker = _Redo([])
    assert await routes._redo_stalled(_request(worker), [prepared], results, "parakeet-v3:fp32") == results
    assert worker.pieces == [] and prepared.windows == _samples([(0.0, 13.0), (7.0, 20.0)])
