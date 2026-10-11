"""Words Parakeet makes up where its input ends (#110), and that only those go."""
from __future__ import annotations

import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import UploadFile

from parakeet_service import routes
from parakeet_service.config import TARGET_SR

# The book's line, as the full-length decode hears it (word, start in s).
SAID = [("smelling", 0.4), ("salts?", 0.9), ("What", 2.2), ("happened?", 2.5), ("he", 3.3), ("muttered.", 3.5)]
END = 4.5  # the clip ends half a second into the pause after "muttered"


class _Parakeet:
    """Hears SAID in the audio it is given, a piece of `wav` (_clip), found by
    its samples, as far as it runs before anything played after it. As its
    input ends at `made_up_at` s, and only then, it makes up `made_up` just
    before the end, as #110's clip did. A piece in `lost` comes back with
    nothing heard at all."""

    def __init__(self, wav, made_up=("and",), made_up_at=END, said=SAID, lost=()):
        self.wav, self.made_up, self.made_up_at, self.said, self.lost = wav, made_up, made_up_at, said, lost
        self.at = {wav[i : i + 8].tobytes(): i for i in range(wav.size - 8)}
        self.pieces = []

    def found(self, piece):
        """Where `piece` starts in `wav`, and where the audio from there ends."""
        start = self.at[piece[:8].tobytes()]
        same = piece[: self.wav.size - start] == self.wav[start : start + piece.size]
        return start, start + (int(np.argmin(same)) if not same.all() else same.size)

    def hear(self, piece):
        self.pieces.append(piece)
        start, stop = self.found(piece)
        if len(self.pieces) in self.lost:
            return SimpleNamespace(text="", tokens=[], timestamps=[])
        words = [(word, at) for word, at in self.said if start <= at * TARGET_SR < stop]
        if stop == int(self.made_up_at * TARGET_SR) and piece.size == stop - start:
            words += [(word, self.made_up_at - 0.48 + 0.08 * i) for i, word in enumerate(self.made_up)]
        tokens = [" " + word for word, _at in words]
        return SimpleNamespace(
            text=routes._decoded(tokens), tokens=tokens, timestamps=[round(at - start / TARGET_SR, 6) for _word, at in words]
        )

    async def submit_many(self, pieces, _model_key):
        return [self.hear(piece) for piece in pieces]


def _clip(seconds=END, ranges=None, windows=None, speech=((0.3, 4.0),)):
    """Noise, loud where there is `speech` and 50 dB quieter elsewhere."""
    rng = np.random.default_rng(110)
    level = np.full(int(seconds * TARGET_SR), 0.001, dtype=np.float32)
    for a, b in speech:
        level[int(a * TARGET_SR) : int(b * TARGET_SR)] = 0.3
    wav = (rng.uniform(-1, 1, level.size) * level).astype(np.float32)
    ranges = [(int(a * TARGET_SR), int(b * TARGET_SR)) for a, b in ranges or [(0.0, seconds)]]
    windows = [(int(a * TARGET_SR), int(b * TARGET_SR)) for a, b in windows] if windows else list(ranges)
    return routes._PreparedAudio(
        waveform=wav,
        ranges=ranges,
        windows=windows,
        speech=None,
        pieces=[wav[a:b] for a, b in windows],
        duration=seconds,
    )


def _request(worker):
    state = SimpleNamespace(worker=worker, ready=True, audio_pool=None)  # None: asyncio's own pool
    return SimpleNamespace(app=SimpleNamespace(state=state))


def _words(result):
    return [word for word, _first, _last in routes._word_spans(routes._extract(result)["tokens"])]


async def _checked(prepared, model="parakeet-v3:int8", fake=_Parakeet, **heard):
    """`prepared`'s pieces decoded and checked, and the fake that heard them."""
    worker = fake(prepared.waveform, **heard)
    results = await worker.submit_many(prepared.pieces, model)
    return await routes._drop_made_up_ends(_request(worker), [prepared], results, model), worker


@pytest.mark.asyncio
async def test_issue_110_a_word_made_up_as_a_clip_ends_in_a_pause_is_dropped():
    [result], worker = await _checked(_clip())
    assert _words(result) == [word for word, _at in SAID]
    assert result.text == "smelling salts? What happened? he muttered."
    # Its ending, here all of it, was heard again twice: a little shorter, and
    # with the pause longer.
    assert [(*worker.found(piece), piece.size) for piece in worker.pieces[1:]] == [
        (0, int(4.3 * TARGET_SR), int(4.3 * TARGET_SR)),
        (0, int(4.5 * TARGET_SR), int(4.8 * TARGET_SR)),
    ]


@pytest.mark.asyncio
async def test_a_run_of_made_up_words_goes_with_the_last():
    [result], _worker = await _checked(_clip(), made_up=["the", "wheel", "of", "the", "wheel."])
    assert _words(result) == [word for word, _at in SAID]


@pytest.mark.asyncio
async def test_a_word_that_was_said_stays_however_late_it_is_timed():
    # The last word said sits at the very end, where made-up words are timed.
    said = SAID + [("Yes.", END - 0.3)]
    [result], _worker = await _checked(_clip(), made_up=(), said=said)
    assert _words(result) == [word for word, _at in said]


@pytest.mark.asyncio
async def test_a_clip_ending_inside_speech_is_not_checked():
    # Cut in its last word, both decodes of its ending would lose that word.
    [result], worker = await _checked(_clip(speech=[(0.3, END)]))
    assert _words(result)[-1] == "and" and len(worker.pieces) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("lost", [{2}, {3}, {2, 3}])
async def test_a_decode_that_loses_the_speech_is_no_sign(lost):
    # Parakeet can return nothing for a whole clip, depending on where it ends:
    # a decode of the ending that did not hear "muttered" says nothing of "and".
    [result], _worker = await _checked(_clip(), lost=lost)
    assert _words(result)[-1] == "and"


@pytest.mark.asyncio
async def test_a_word_one_decode_of_the_ending_hears_stays():
    class _Stubborn(_Parakeet):
        def hear(self, piece):
            result = super().hear(piece)
            if len(self.pieces) == 3:  # the longer decode hears "and" too
                result = SimpleNamespace(
                    text=result.text + " and",
                    tokens=result.tokens + [" and"],
                    timestamps=result.timestamps + [END - 0.4 - self.found(piece)[0] / TARGET_SR],
                )
            return result

    [result], _worker = await _checked(_clip(), fake=_Stubborn)
    assert _words(result)[-1] == "and"


@pytest.mark.asyncio
async def test_a_word_heard_as_another_at_the_same_spot_stays():
    # #112's review: "...eat some of the other bit" came back "bed." from the
    # clip, and "bit" from a decode of its ending: speech there, so it stays.
    said = SAID[:-1] + [("muttered", 3.5), ("bed.", 3.9)]

    class _Hard(_Parakeet):
        def hear(self, piece):
            result = super().hear(piece)
            if len(self.pieces) > 1:  # the decodes of its ending hear it another way
                tokens = [" bit." if token == " bed." else token for token in result.tokens]
                result = SimpleNamespace(text=routes._decoded(tokens), tokens=tokens, timestamps=result.timestamps)
            return result

    [result], worker = await _checked(_clip(), fake=_Hard, made_up=(), said=said)
    assert len(worker.pieces) == 3 and _words(result)[-1] == "bed."


def test_a_word_heard_another_way_is_the_same_word():
    at = 3 * TARGET_SR
    assert routes._heard_in([("rat", at - 1000), ("hole.", at + 2000)], "rat-hole.", at)
    assert routes._heard_in([("Lori,", at + 4000)], "Lorry.", at)
    assert routes._heard_in([("A", at - 8000), ("long", at - 4000), ("TAL", at)], "Tail.", at)
    assert routes._heard_in([("Chapter", at - 8000), ("three", at)], "III", at)
    assert routes._heard_in([("3rd", at)], "third.", at)
    assert not routes._heard_in([("he", at)], "the", at)  # short words must match
    assert routes._heard_in([("the", at)], "'", at)  # a lone mark: nothing to tell by
    assert not routes._heard_in([("or", at)], "and", at)
    assert not routes._heard_in([("and", at + 2 * TARGET_SR)], "and", at)  # another word, later


@pytest.mark.asyncio
async def test_only_words_near_the_end_are_checked():
    # A clip ending 3 s into the pause: "muttered" is 4 s before its end,
    # and nothing is decoded again.
    said = SAID[:-1] + [("muttered.", 2.0)]
    [result], worker = await _checked(_clip(6.0, speech=[(0.3, 2.6)]), made_up=(), said=said)
    assert len(worker.pieces) == 1 and _words(result)[-1] == "muttered."


@pytest.mark.asyncio
async def test_a_piece_that_decodes_context_past_its_end_is_not_checked():
    # The words past its range are the next piece's: _trimmed drops them.
    prepared = _clip(10.0, ranges=[(0.0, 4.5), (4.5, 10.0)], windows=[(0.0, 6.0), (4.0, 10.0)])
    results, worker = await _checked(prepared, made_up_at=6.0)
    assert len(worker.pieces) == 2  # neither: the last piece's final word is long before its end
    assert _words(results[0])[-1] == "and"


@pytest.mark.asyncio
async def test_the_last_piece_of_long_audio_is_checked():
    said = SAID + [("You", 6.0), ("had", 6.3), ("a", 6.5), ("seizure.", 6.6), ("Yes.", 9.0)]
    prepared = _clip(10.0, ranges=[(0.0, 5.0), (5.0, 10.0)], windows=[(0.0, 7.0), (3.0, 10.0)], speech=[(0.3, 9.4)])
    results, worker = await _checked(prepared, made_up_at=10.0, said=said)
    assert _words(results[1])[-1] == "Yes."
    assert len(worker.pieces) == 4  # the first piece's ending is the second's to hear


@pytest.mark.asyncio
async def test_whisper_is_never_checked():
    [result], worker = await _checked(_clip(), model="whisper-base:fp32")
    assert _words(result)[-1] == "and" and len(worker.pieces) == 1


@pytest.mark.asyncio
async def test_a_failed_decode_of_the_ending_keeps_what_was_heard(caplog):
    class _Failing(_Parakeet):
        async def submit_many(self, pieces, model_key):
            if self.pieces:
                raise RuntimeError("out of memory")
            return await super().submit_many(pieces, model_key)

    [result], _worker = await _checked(_clip(), fake=_Failing)
    assert _words(result)[-1] == "and"
    assert "endings again failed" in caplog.text


def _state(worker):
    return SimpleNamespace(
        worker=worker,
        ready=True,
        audio_pool=ThreadPoolExecutor(max_workers=2, thread_name_prefix="audio"),
        align_pool=ThreadPoolExecutor(max_workers=1, thread_name_prefix="align"),
    )


@pytest.mark.asyncio
async def test_issue_110_a_transcription_has_no_made_up_last_word(monkeypatch):
    clip = _clip()
    monkeypatch.setattr(routes, "_prepare_audio", lambda *_args: clip)
    state = _state(_Parakeet(clip.waveform))
    try:
        response = await routes.transcribe(
            request=SimpleNamespace(app=SimpleNamespace(state=state)),
            file=UploadFile(io.BytesIO(b"clip"), filename="a.wav"),
            model="parakeet-v3",
            quantization="int8",
            response_format="verbose_json",
            timestamp_granularities=["word"],
            timestamp_granularities_plain=None,
            language=None,
            prompt=None,
            temperature=None,
            spoken_numbers=None,
            aligner_name=None,
            retime_words=None,
            vad_filter=None,
        )
    finally:
        state.audio_pool.shutdown()
        state.align_pool.shutdown()
    body = json.loads(response.body)
    assert body["text"] == "smelling salts? What happened? he muttered."
    assert [word["word"] for word in body["words"]] == [word for word, _at in SAID]


@pytest.mark.asyncio
async def test_issue_110_a_batch_has_no_made_up_last_word(monkeypatch):
    clip = _clip()
    monkeypatch.setattr(routes, "_prepare_audio", lambda *_args: clip)
    state = _state(_Parakeet(clip.waveform))
    try:
        body = await routes.transcribe_batch(
            request=SimpleNamespace(app=SimpleNamespace(state=state)),
            files=[UploadFile(io.BytesIO(b"clip"), filename=f"{i}.wav") for i in range(2)],
            model="parakeet-v3",
            quantization="int8",
            spoken_numbers=None,
            aligner_name=None,
            vad_filter=None,
        )
    finally:
        state.audio_pool.shutdown()
        state.align_pool.shutdown()
    assert [item["text"] for item in body["results"]] == ["smelling salts? What happened? he muttered."] * 2


@pytest.mark.asyncio
async def test_long_audio_is_never_scanned_on_the_event_loop(monkeypatch):
    threads = []
    ending, made_up_end = routes._ending, routes._made_up_end

    def noting(work):
        def run(*args):
            threads.append(threading.current_thread().name)
            return work(*args)

        return run

    monkeypatch.setattr(routes, "_ending", noting(ending))
    monkeypatch.setattr(routes, "_made_up_end", noting(made_up_end))
    said = SAID + [("You", 6.0), ("had", 6.3), ("a", 6.5), ("seizure.", 6.6), ("Yes.", 9.0)]
    prepared = _clip(10.0, ranges=[(0.0, 5.0), (5.0, 10.0)], windows=[(0.0, 7.0), (3.0, 10.0)], speech=[(0.3, 9.4)])
    worker = _Parakeet(prepared.waveform, made_up_at=10.0, said=said)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio")
    try:
        request = _request(worker)
        request.app.state.audio_pool = pool
        results = await worker.submit_many(prepared.pieces, "parakeet-v3:int8")
        results = await routes._drop_made_up_ends(request, [prepared], results, "parakeet-v3:int8")
    finally:
        pool.shutdown()
    assert _words(results[1])[-1] == "Yes."
    assert threads and all(name.startswith("audio") for name in threads)


@pytest.mark.asyncio
async def test_parakeet_check_endings_false_turns_it_off(monkeypatch):
    monkeypatch.setattr(routes, "CHECK_ENDINGS", False)
    [result], worker = await _checked(_clip())
    assert _words(result)[-1] == "and" and len(worker.pieces) == 1
