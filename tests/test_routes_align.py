"""/v1/audio/alignments: known text in, each word's times and score out, with
the audio decoder and the aligner faked.

The handler is called directly (CI has no httpx for TestClient); everything
between the form fields and the response body is the real code.
"""
from __future__ import annotations

import io
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile

from parakeet_service import aligner, routes, schemas
from parakeet_service.config import TARGET_SR

BASE = "wav2vec2-base-960h"
OMNI = "omnilingual-ctc-300m"


@pytest.fixture
def heard(monkeypatch):
    """Two seconds of audio. The fake aligner times word i at 0.5 s + 0.6 s each,
    scores it 0.9, and leaves a word without letters untimed; calls are recorded."""
    calls = []

    class FakeChunk:
        def __init__(self, wav, language, choice):
            self.wav, self.language, self.choice = wav, language, choice

        def words(self, words):
            calls.append({
                "words": list(words), "language": self.language, "aligner": self.choice,
                "samples": self.wav.size, "thread": threading.current_thread().name,
            })
            return [
                None if not any(c.isalpha() for c in word) else (0.5 + 0.6 * i, 0.9 + 0.6 * i, 0.9)
                for i, word in enumerate(words)
            ]

    def fake_for_chunk(wav, language, name=None, quantization=None):
        return FakeChunk(wav, language, (name, quantization)) if name and aligner.aligns(name, language) else None

    monkeypatch.setattr(aligner, "for_chunk", fake_for_chunk)
    monkeypatch.setattr(aligner, "ALIGN_DEFAULT_LANGUAGE", "en")
    def decode(raw, max_seconds=None):
        seconds = float(raw.decode())
        if max_seconds is not None:  # as load_audio: stops a second past the limit
            seconds = min(seconds, max_seconds + 1.0)
        return np.zeros(int(seconds * TARGET_SR), np.float32)

    monkeypatch.setattr(routes, "load_audio", decode)
    return calls


async def _align(text="hello world", aligner_name=BASE, language=None, seconds=2.0):
    state = SimpleNamespace(
        audio_pool=ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio"),
        align_pool=ThreadPoolExecutor(max_workers=1, thread_name_prefix="align"),
    )
    try:
        return await routes.align(
            request=SimpleNamespace(app=SimpleNamespace(state=state)),
            file=UploadFile(io.BytesIO(str(seconds).encode()), filename="a.wav"),
            text=text,
            aligner_name=aligner_name,
            language=language,
        )
    finally:
        state.audio_pool.shutdown()
        state.align_pool.shutdown()


async def _status(**kwargs):
    with pytest.raises(HTTPException) as raised:
        await _align(**kwargs)
    return raised.value.status_code, raised.value.detail


@pytest.mark.asyncio
async def test_each_word_of_the_text_gets_times_and_a_score(heard):
    body = await _align("My love, — still", language="en")
    schemas.Alignment.model_validate(body)
    assert body["aligner"] == f"{BASE}:int8" and body["language"] == "en" and body["duration"] == 2.0
    assert body["words"] == [
        {"word": "My", "start": 0.5, "end": 0.9, "score": 0.9},
        {"word": "love,", "start": pytest.approx(1.1), "end": pytest.approx(1.5), "score": 0.9},
        {"word": "—", "start": None, "end": None, "score": None},  # nothing the aligner can say
        # past the end of the audio (the last 20 ms frame can overhang it)
        {"word": "still", "start": 2.0, "end": 2.0, "score": 0.9},
    ]
    assert heard[0]["words"] == ["My", "love,", "—", "still"]


@pytest.mark.asyncio
async def test_alignment_runs_on_the_align_pool(heard):
    await _align()
    assert heard[0]["thread"].startswith("align")
    assert heard[0]["samples"] == 2 * TARGET_SR


@pytest.mark.asyncio
async def test_the_aligner_and_its_precision_are_the_request_s(heard):
    body = await _align(aligner_name=f"{OMNI}:fp32", language="fr", text="bonjour")
    assert body["aligner"] == f"{OMNI}:fp32" and body["language"] == "fr"
    assert heard[0]["aligner"] == (OMNI, "fp32") and heard[0]["language"] == "fr"


@pytest.mark.asyncio
async def test_without_language_the_text_is_aligned_as_the_default(heard, monkeypatch):
    assert (await _align(language="auto"))["language"] == "en"
    monkeypatch.setattr(aligner, "ALIGN_DEFAULT_LANGUAGE", "")
    assert await _status() == (400, "Send `language`: this server has no default language to align")
    assert not heard[1:]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "complaint"),
    [
        ({"text": "  \n "}, "No text provided"),
        ({"aligner_name": "wav2vec2"}, "Unknown aligner"),
        ({"aligner_name": f"{BASE}:fp8"}, "has no 'fp8' quantization"),
        ({"language": "fr"}, "does not align 'fr'"),
        ({"language": "English"}, "ISO 639-1"),
    ],
)
async def test_a_bad_request_is_a_400_before_any_audio_is_heard(heard, kwargs, complaint):
    status, detail = await _status(**kwargs)
    assert status == 400 and complaint in detail
    assert not heard


@pytest.mark.asyncio
async def test_text_the_aligner_cannot_align_is_a_400_saying_why(heard, monkeypatch):
    def refuse(_wav, _language, *_choice):
        return SimpleNamespace(words=lambda _words: (_ for _ in ()).throw(aligner.AlignmentError("too short")))

    monkeypatch.setattr(aligner, "for_chunk", refuse)
    assert await _status() == (400, "Can't align the text: too short")


@pytest.mark.asyncio
async def test_an_aligner_that_cannot_load_is_a_503_naming_it(heard, monkeypatch):
    monkeypatch.setattr(aligner, "for_chunk", lambda *_: None)
    assert await _status() == (503, f"Aligner '{BASE}:int8' could not be loaded")


@pytest.mark.asyncio
async def test_audio_over_the_alignment_limit_is_a_413(heard, monkeypatch):
    monkeypatch.setattr(routes, "ALIGN_MAX_AUDIO_SECONDS", 60.0)
    limits = []
    decode = routes.load_audio
    monkeypatch.setattr(routes, "load_audio", lambda raw, max_seconds=None: limits.append(max_seconds) or decode(raw, max_seconds))
    for seconds in (3600.0, 60.5):
        assert await _status(seconds=seconds) == (413, "audio duration exceeds the alignment limit 60.0s")
    assert not heard
    assert limits == [60.0, 60.0]  # never decoded far past the limit
    assert (await _align(seconds=60.0))["duration"] == 60.0


@pytest.mark.asyncio
async def test_audio_that_cannot_be_decoded_is_a_415(heard, monkeypatch):
    monkeypatch.setattr(routes, "load_audio", lambda _raw, _max=None: 1 / 0)
    assert await _status() == (415, "Audio could not be decoded")
    monkeypatch.setattr(routes, "load_audio", lambda _raw, _max=None: np.zeros(0, np.float32))
    assert await _status() == (415, "Audio could not be decoded")


@pytest.mark.asyncio
async def test_a_real_chunk_aligner_answers_the_route(heard, monkeypatch):
    # the real ChunkAligner.words over a fake wav2vec2 that hears only blank:
    # the text can't fit in no frames, which is a 400, not a 500
    vocab = {"<pad>": 0, "|": 4, "'": 5, **{chr(ord("A") + i): 6 + i for i in range(26)}}

    class Silent:
        def get_inputs(self):
            return [SimpleNamespace(name="input_values")]

        def run(self, _outputs, feeds):
            frames = (feeds["input_values"].shape[1] - aligner._MIN_SAMPLES) // aligner._STRIDE + 1
            out = np.zeros((1, frames, len(vocab) + 2), np.float32)
            out[..., 0] = 8.0
            return [out]

    monkeypatch.setattr(aligner, "for_chunk", lambda wav, *_: aligner.ChunkAligner(
        wav, Silent(), vocab, {"blank": "<pad>", "separator": "|", "normalisers": ["english", "upper"]}
    ))
    body = await _align("hi")  # a word fits; the model hears none of it
    assert body["words"][0]["score"] < 0.01
    status, detail = await _status(text="word " * 200)
    assert status == 400 and "too short" in detail
