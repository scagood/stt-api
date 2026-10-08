"""The /v1/audio/transcriptions word-timestamp and spoken-number paths (and the
batch endpoint), with Parakeet and the aligner faked.

The handlers are called directly (CI has no httpx for TestClient); everything
between the form fields and the response body is the real code. The fake
audio's bytes are the transcript Parakeet "hears".
"""
from __future__ import annotations

import inspect
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, UploadFile, params
from starlette.datastructures import FormData

from parakeet_service import aligner, routes, schemas
from parakeet_service.config import TARGET_SR
from parakeet_service.model import get_model

WORDS = ["hello", "world"]


class _Worker:
    async def submit_many(self, pieces, _model_name):
        return [
            SimpleNamespace(
                text=piece,
                tokens=[" " + word for word in piece.split()],
                timestamps=[0.8 * i for i in range(len(piece.split()))],
            )
            for piece in pieces
        ]


def _prepared(raw, *_bounds):
    return routes._PreparedAudio(
        waveform=None, ranges=[(0, 2 * TARGET_SR)], windows=[(0, 2 * TARGET_SR)], speech=[],
        pieces=[raw.decode()], duration=2.0,
    )


@pytest.fixture
def calls(monkeypatch):
    """Record aligner calls; the fake aligner re-times words to 0.1 s + 1.5 s each
    and hears every reading equally well (so the first is kept)."""
    recorded = []

    class FakeChunk:
        def __init__(self, language, aligner):
            self.language, self.aligner = language, aligner

        def spans(self, words):
            recorded.append({
                "words": list(words), "language": self.language, "aligner": self.aligner,
                "thread": threading.current_thread().name,
            })
            return [(0.1 + 1.5 * i, 0.4 + 1.5 * i) for i in range(len(words))]

        def scores(self, options, _start, _end):
            recorded.append({"options": list(options), "thread": threading.current_thread().name})
            return [0.0] * len(options)

        def best(self, _options, _start, _end):
            return 0

    def fake_for_chunk(_wav, language, name=None, quantization=None):
        return FakeChunk(language, (name, quantization)) if name and aligner.aligns(name, language) else None

    async def fake_prepare(_request, raw, *_bounds):
        return _prepared(raw)

    monkeypatch.setattr(aligner, "for_chunk", fake_for_chunk)
    monkeypatch.setattr(aligner, "ALIGN_DEFAULT_LANGUAGE", "en")
    monkeypatch.setattr(routes, "_prepare_in_pool", fake_prepare)
    monkeypatch.setattr(routes, "_prepare_audio", _prepared)
    return recorded


def _pool(thread):
    """Where a thread belongs: "align" or "audio" pool, else "inline" (the event loop)."""
    return next((pool for pool in ("align", "audio") if thread.startswith(pool)), "inline")


@pytest.fixture
def stitched(monkeypatch):
    """Where each _stitch ran: "align", "audio" or "inline" (see _pool)."""
    threads = []
    stitch = routes._stitch

    def recording(*args, **kwargs):
        threads.append(_pool(threading.current_thread().name))
        return stitch(*args, **kwargs)

    monkeypatch.setattr(routes, "_stitch", recording)
    return threads


@pytest.fixture
def speak(monkeypatch):
    monkeypatch.setattr(routes, "SPOKEN_NUMBERS", True)


def _state():
    return SimpleNamespace(
        worker=_Worker(),
        ready=True,
        audio_pool=ThreadPoolExecutor(max_workers=2, thread_name_prefix="audio"),
        align_pool=ThreadPoolExecutor(max_workers=1, thread_name_prefix="align"),
    )


async def _transcribe(
    response_format="verbose_json",
    granularity="word",
    language=None,
    text="hello world",
    aligner_name=None,
    spoken_numbers=None,
    model="parakeet-v3",
    quantization=None,
    retime_words=None,
):
    state = _state()
    try:
        response = await routes.transcribe(
            request=SimpleNamespace(app=SimpleNamespace(state=state)),
            file=UploadFile(io.BytesIO(text.encode()), filename="a.wav"),
            model=model,
            quantization=quantization,
            response_format=response_format,
            timestamp_granularities=[granularity] if granularity else None,
            timestamp_granularities_plain=None,
            language=language,
            prompt=None,
            temperature=None,
            spoken_numbers=spoken_numbers,
            aligner_name=aligner_name,
            retime_words=retime_words,
        )
    finally:
        state.audio_pool.shutdown()
        state.align_pool.shutdown()
    return json.loads(response.body) if response_format.endswith("json") else response.body.decode()


async def _batch_body(*texts, spoken_numbers=None, aligner_name=None, model="parakeet-v3"):
    state = _state()
    try:
        body = await routes.transcribe_batch(
            request=SimpleNamespace(app=SimpleNamespace(state=state)),
            files=[UploadFile(io.BytesIO(text.encode()), filename=f"{i}.wav") for i, text in enumerate(texts)],
            model=model,
            quantization=None,
            spoken_numbers=spoken_numbers,
            aligner_name=aligner_name,
        )
    finally:
        state.audio_pool.shutdown()
        state.align_pool.shutdown()
    return body


async def _batch(*texts, **kwargs):
    return [item["text"] for item in (await _batch_body(*texts, **kwargs))["results"]]


BASE = "wav2vec2-base-960h"


@pytest.mark.asyncio
async def test_word_request_is_aligned_on_the_align_pool(calls):
    body = await _transcribe(language="en", aligner_name=BASE)
    assert [c["words"] for c in calls] == [WORDS]
    assert calls[0]["language"] == "en"
    assert calls[0]["aligner"] == (BASE, "int8")  # its default_quantization
    assert calls[0]["thread"].startswith("align")  # not the audio pool, not the loop
    assert [(w["word"], w["start"], w["end"]) for w in body["words"]] == [
        ("hello", 0.1, 0.4),
        ("world", 1.6, 1.9),
    ]
    assert body["language"] == "en"
    # the segment was widened to cover the re-timed last word
    assert body["segments"][0]["end"] >= body["words"][-1]["end"]


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "ja"])
async def test_without_an_aligner_words_keep_model_times(calls, language):
    body = await _transcribe(language=language)
    assert calls == []
    assert body["words"][0]["start"] == 0.0 and body["words"][1]["start"] == 0.8
    assert body["language"] == language


@pytest.mark.asyncio
async def test_with_no_default_language_an_aligner_needs_language(calls, monkeypatch):
    # PARAKEET_ALIGN_DEFAULT_LANGUAGE empty: align only when `language` is sent, never a 400
    monkeypatch.setattr(aligner, "ALIGN_DEFAULT_LANGUAGE", "")
    body = await _transcribe(language=None, aligner_name=BASE)
    assert calls == [] and body["words"][1]["start"] == 0.8
    assert await _batch("hello", aligner_name=BASE) == ["hello"]
    await _transcribe(language="en", aligner_name=BASE)
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en-GB", "en_GB", "EN", "English", "de-DE"])
async def test_language_is_a_bare_iso_code_on_every_request(calls, language):
    for response_format in ("json", "verbose_json"):  # whether or not anything uses it
        with pytest.raises(HTTPException) as caught:
            await _transcribe(response_format=response_format, language=language)
        assert caught.value.status_code == 400 and "ISO 639-1" in caught.value.detail


@pytest.mark.asyncio
async def test_the_aligners_language_is_checked_only_for_word_times(calls):
    # json: the aligner would never run for French (spoken numbers are English)
    body = await _transcribe(response_format="json", language="fr", aligner_name=BASE)
    assert body["text"] == "hello world" and calls == []
    with pytest.raises(HTTPException, match="does not align 'fr'"):
        await _transcribe(language="fr", aligner_name=BASE)


@pytest.mark.asyncio
async def test_model_and_aligner_take_a_quantization_after_a_colon(calls, monkeypatch):
    keys = []
    submit = _Worker.submit_many

    async def recording(self, pieces, model_key):
        keys.append(model_key)
        return await submit(self, pieces, model_key)

    monkeypatch.setattr(_Worker, "submit_many", recording)
    await _transcribe(model="parakeet-v3:int8", language="en", aligner_name=f"{BASE}:fp32")
    assert keys == ["parakeet-v3:int8"]
    assert calls[0]["aligner"] == (BASE, "fp32")
    assert await _batch("hello", model="parakeet-v3:int8", aligner_name=f"{BASE}:fp32") == ["hello"]
    assert keys[-1] == "parakeet-v3:int8"


@pytest.mark.asyncio
async def test_a_colon_and_the_quantization_field_must_agree(calls):
    with pytest.raises(HTTPException) as caught:
        await _transcribe(language="en", model="parakeet-v3:fp16", quantization="int8")
    assert caught.value.status_code == 400 and "quantization says 'int8'" in caught.value.detail


@pytest.mark.asyncio
async def test_aligner_quantization_is_a_400_pointing_at_the_colon():
    # FastAPI drops a form field it doesn't know: without this, its precision would be ignored
    async def form(*pairs):
        return FormData(list(pairs))

    sent = SimpleNamespace(form=lambda: form(("aligner", BASE), ("aligner_quantization", "fp32")))
    with pytest.raises(HTTPException) as caught:
        await routes._no_aligner_quantization(sent)
    assert caught.value.status_code == 400 and f"aligner={BASE}:fp32" in caught.value.detail
    assert await routes._no_aligner_quantization(SimpleNamespace(form=lambda: form(("aligner", BASE)))) is None
    checked = {
        route.path for route in routes.router.routes
        if any(depends.dependency is routes._no_aligner_quantization for depends in getattr(route, "dependencies", []))
    }
    assert checked == {"/v1/audio/transcriptions", "/v1/audio/transcriptions/batch"}


@pytest.mark.asyncio
async def test_a_model_that_cannot_load_is_a_503_naming_it(calls, monkeypatch):
    def unavailable():  # the first step of a cold load
        raise RuntimeError("PARAKEET_USE_GPU=true but CUDAExecutionProvider is unavailable")

    async def loading(self, pieces, model_key):  # the workers' up-front load
        return [get_model(model_key)]

    monkeypatch.setattr("parakeet_service.model._resolve_providers", unavailable)
    monkeypatch.setattr(_Worker, "submit_many", loading)
    for request in (_transcribe(model="parakeet-v2:fp16"), _batch("hello", model="parakeet-v2:fp16")):
        with pytest.raises(HTTPException) as caught:
            await request
        assert caught.value.status_code == 503
        assert caught.value.detail == (
            "Model 'parakeet-v2:fp16' could not be loaded: "
            "RuntimeError: PARAKEET_USE_GPU=true but CUDAExecutionProvider is unavailable"
        )


@pytest.mark.asyncio
async def test_parakeet_v3_languages_are_aligned_in_their_own_language(calls):
    body = await _transcribe(language="fr", aligner_name="Omnilingual-CTC-300M")
    assert [(c["language"], c["aligner"]) for c in calls] == [("fr", ("omnilingual-ctc-300m", "int8"))]
    assert body["words"][1]["start"] == 1.6


@pytest.mark.asyncio
async def test_missing_language_reports_auto_and_uses_the_default(calls):
    body = await _transcribe(language=None, aligner_name=BASE)
    assert len(calls) == 1 and body["language"] == "auto"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("aligner_name", "language", "complaint"),
    [
        ("wav2vec3", "en", "Unknown aligner 'wav2vec3'"),
        (f"{BASE}:fp8", "en", "no 'fp8' quantization"),
        (f"{BASE}:", "en", "no quantization after ':'"),
        (BASE, "fr", "does not align 'fr'"),
    ],
)
async def test_a_bad_aligner_is_a_400_naming_the_choices(calls, aligner_name, language, complaint):
    with pytest.raises(HTTPException) as caught:
        await _transcribe(language=language, aligner_name=aligner_name)
    assert caught.value.status_code == 400 and complaint in caught.value.detail
    if language == "en":  # the batch endpoint takes no `language`: the default, English
        with pytest.raises(HTTPException, match="400"):
            await _batch("hello", aligner_name=aligner_name)


@pytest.mark.asyncio
@pytest.mark.parametrize("response_format", ["verbose_json", "json"])
@pytest.mark.parametrize("granularity", ["char", "Word", "words"])
async def test_an_unknown_granularity_is_a_400(calls, response_format, granularity):
    with pytest.raises(HTTPException) as caught:
        await _transcribe(response_format=response_format, granularity=granularity)
    assert caught.value.status_code == 400 and repr(granularity) in caught.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_format", "granularity"),
    [("verbose_json", None), ("verbose_json", "segment"), ("json", "word"), ("srt", "word")],
)
async def test_alignment_only_runs_when_words_are_returned(calls, stitched, response_format, granularity):
    await _transcribe(response_format=response_format, granularity=granularity, aligner_name=BASE)
    assert calls == []
    assert stitched == ["inline"]  # nothing to align: no queue


@pytest.mark.parametrize(
    ("handler", "name", "annotation"),
    [
        (routes.transcribe, "spoken_numbers", ("Optional[bool]", "bool | None")),
        (routes.transcribe_batch, "spoken_numbers", ("Optional[bool]", "bool | None")),
        (routes.transcribe, "aligner_name", ("Optional[str]", "str | None")),
        (routes.transcribe_batch, "aligner_name", ("Optional[str]", "str | None")),
        (routes.transcribe, "retime_words", ("Optional[bool]", "bool | None")),
    ],
)
def test_the_switches_are_optional_form_fields(handler, name, annotation):
    # the handlers are called directly here, so pin what FastAPI will parse
    field = inspect.signature(handler).parameters[name]
    assert isinstance(field.default, params.Form) and field.default.default is None
    assert field.annotation in annotation
    assert "align_words" not in inspect.signature(handler).parameters  # an aligner is named, or none


def test_timestamp_granularities_is_documented_once():
    app = FastAPI()
    app.include_router(routes.router)
    schema = app.openapi()
    body = schema["paths"]["/v1/audio/transcriptions"]["post"]["requestBody"]
    name = body["content"]["multipart/form-data"]["schema"]["$ref"].rsplit("/", 1)[1]
    fields = schema["components"]["schemas"][name]["properties"]
    assert len([field for field in fields if field.startswith("timestamp_granularities")]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_format", "granularity", "aligner_name", "schema"),
    [
        ("json", None, None, schemas.Transcription),
        ("verbose_json", None, None, schemas.VerboseTranscription),
        ("verbose_json", "word", None, schemas.VerboseTranscription),
        ("verbose_json", "word", BASE, schemas.VerboseTranscription),
    ],
)
async def test_json_responses_match_the_documented_schema(calls, response_format, granularity, aligner_name, schema):
    body = await _transcribe(response_format, granularity, language="en", aligner_name=aligner_name)
    schema.model_validate(body)
    assert (body.get("words") is not None) == (granularity == "word")


@pytest.mark.asyncio
async def test_batch_response_matches_the_documented_schema(calls):
    schemas.BatchTranscription.model_validate(await _batch_body("hello world", "goodbye"))


@pytest.mark.asyncio
async def test_timestamp_granularities_without_brackets_is_read_from_the_form():
    async def form(*pairs):
        return FormData(list(pairs))

    plain = SimpleNamespace(form=lambda: form(
        ("timestamp_granularities", "word"),
        ("timestamp_granularities", "segment"),
        ("timestamp_granularities[]", "ignored"),
    ))
    assert await routes._plain_granularities(plain) == ["word", "segment"]
    assert await routes._plain_granularities(SimpleNamespace(form=form)) is None


def test_aligners_are_listed_like_models():
    listing = routes.list_aligners()
    assert [card["id"] for card in listing["data"]] == list(routes.ALIGNER_CONFIGS)
    card = routes.retrieve_aligner("Omnilingual-CTC-300M")
    assert card["object"] == "aligner" and card["default_quantization"] in card["quantizations"]
    assert "fr" in card["language"]
    with pytest.raises(HTTPException) as caught:
        routes.retrieve_aligner("nope")
    assert caught.value.status_code == 404


def test_health_reports_aligner_state(monkeypatch):
    monkeypatch.setattr(aligner, "status", lambda: {"wav2vec2-base-960h:int8": "failed"})
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(ready=True)))
    assert routes.health(request)["aligner"] == {"wav2vec2-base-960h:int8": "failed"}


# --------------------------------------------------------------------------- #
# Spoken numbers (PARAKEET_SPOKEN_NUMBERS)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_spoken_numbers_are_off_by_default(calls):
    body = await _transcribe(response_format="json", text="It cost $5 today.")
    assert body["text"] == "It cost $5 today."
    assert await _batch("It cost $5 today.") == ["It cost $5 today."]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server_default", "spoken_numbers", "said"),
    [(False, None, False), (False, True, True), (True, None, True), (True, False, False)],
)
async def test_the_request_says_numbers_or_not_else_the_server_default(
    calls, monkeypatch, server_default, spoken_numbers, said
):
    monkeypatch.setattr(routes, "SPOKEN_NUMBERS", server_default)
    text = "It cost five dollars today." if said else "It cost $5 today."
    body = await _transcribe(response_format="json", text="It cost $5 today.", spoken_numbers=spoken_numbers)
    assert body["text"] == text
    assert await _batch("It cost $5 today.", spoken_numbers=spoken_numbers) == [text]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("language", "text"),
    [
        (None, "It cost five dollars today."),  # PARAKEET_ALIGN_DEFAULT_LANGUAGE
        ("en", "It cost five dollars today."),
        ("fr", "It cost $5 today."),
    ],
)
async def test_spoken_numbers_are_english_only(calls, speak, language, text):
    body = await _transcribe(response_format="json", language=language, text="It cost $5 today.")
    assert body["text"] == text


@pytest.mark.asyncio
async def test_spoken_numbers_follow_the_default_language(calls, speak, monkeypatch):
    monkeypatch.setattr(aligner, "ALIGN_DEFAULT_LANGUAGE", "")
    assert (await _transcribe(response_format="json", text="It cost $5."))["text"] == "It cost $5."
    assert await _batch("It cost $5.") == ["It cost $5."]
    assert (await _transcribe(response_format="json", language="en", text="It cost $5."))["text"] == (
        "It cost five dollars."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response_format", "granularity"),
    [
        ("json", None), ("text", None), ("srt", None), ("vtt", None),
        ("verbose_json", "segment"), ("verbose_json", "word"),
    ],
)
async def test_spoken_numbers_reach_every_response_format(calls, speak, response_format, granularity):
    body = await _transcribe(response_format=response_format, granularity=granularity, text="It cost $5 today.")
    if response_format == "verbose_json":
        assert body["text"] == body["segments"][0]["text"] == "It cost five dollars today."
        if granularity == "word":
            assert [w["word"] for w in body["words"]] == ["It", "cost", "five", "dollars", "today."]
    else:
        body = body["text"] if response_format == "json" else body
        assert "It cost five dollars today." in body and "$5" not in body


@pytest.mark.asyncio
async def test_batch_treats_an_english_only_model_as_english(calls, speak, monkeypatch):
    # as single requests do: parakeet-v2's numbers are English whatever the default
    monkeypatch.setattr(aligner, "ALIGN_DEFAULT_LANGUAGE", "")
    assert await _batch("It cost $5.", model="parakeet-v2") == ["It cost five dollars."]
    assert await _batch("It cost $5.") == ["It cost $5."]  # parakeet-v3: the (empty) default


@pytest.mark.asyncio
async def test_batch_says_numbers_too(calls, speak):
    texts = await _batch("It cost $5 today.", "No numbers here.")
    assert texts == ["It cost five dollars today.", "No numbers here."]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "pool"),
    [
        ("No numbers here.", "inline"),
        ("No numbers in an MP3 here.", "audio"),  # a digit: phrases are looked for off the loop
        ("It opens at 6pm.", "audio"),  # one reading, no rivals: nothing to hear
        ("It opens at 6 pm.", "audio"),  # nor spaced
        ("That'll be £2.10 please.", "align"),
    ],
)
async def test_spoken_numbers_queue_on_the_align_pool_only_to_hear_one(calls, stitched, speak, text, pool):
    await _transcribe(response_format="json", text=text, aligner_name=BASE)
    await _batch(text, aligner_name=BASE)
    assert stitched == [pool, pool]
    assert all(call["thread"].startswith("align") for call in calls)
    assert bool(calls) == (pool == "align")


@pytest.mark.asyncio
async def test_without_an_aligner_spoken_numbers_never_queue_to_hear(calls, stitched, speak):
    text = "That'll be £2.10 please."
    assert (await _transcribe(response_format="json", text=text))["text"] == "That'll be two pounds ten please."
    assert await _batch(text) == ["That'll be two pounds ten please."]
    assert calls == [] and stitched == ["audio", "audio"]


@pytest.mark.asyncio
async def test_a_number_only_in_the_context_does_not_queue_to_hear_it(calls, stitched):
    # "£2.10" starts past the range's end (2 s): the next piece's to keep, and to hear
    prepared = routes._PreparedAudio(
        waveform=None, ranges=[(0, 2 * TARGET_SR)], windows=[(0, 4 * TARGET_SR)], speech=[],
        pieces=["piece"], duration=4.0,
    )
    result = SimpleNamespace(
        text="Hello. That'll be £2.10",
        tokens=[" Hello", ".", " That", "'ll", " be", " ", "£", "2", ".", "1", "0"],
        timestamps=[0.2, 0.5, 2.5, 2.7, 2.9, 3.1, 3.1, 3.2, 3.3, 3.4, 3.5],
    )
    state = _state()
    try:
        text, _segments, _words = await routes._stitch_request(
            SimpleNamespace(app=SimpleNamespace(state=state)), prepared, [result],
            speak=True, language="en", aligner_choice=(BASE, "int8"),
        )
    finally:
        state.audio_pool.shutdown()
        state.align_pool.shutdown()
    assert text == "Hello."
    assert stitched == ["audio"] and calls == []


@pytest.mark.asyncio
async def test_long_audio_is_never_scanned_or_joined_on_the_event_loop(stitched, monkeypatch):
    # Hours of audio are thousands of tokens: finding skipped speech, trimming
    # and joining them run on the audio pool, even for plain text.
    threads = []
    stalled = routes._stalled

    def recording(*args):
        threads.append(_pool(threading.current_thread().name))
        return stalled(*args)

    monkeypatch.setattr(routes, "_stalled", recording)
    prepared = routes._PreparedAudio(
        waveform=None, ranges=[(0, 2 * TARGET_SR), (2 * TARGET_SR, 4 * TARGET_SR)],
        windows=[(0, 3 * TARGET_SR), (TARGET_SR, 4 * TARGET_SR)], speech=[], pieces=["one", "two"], duration=4.0,
    )
    results = [
        SimpleNamespace(text="hello world", tokens=[" hello", " world"], timestamps=[0.5, 2.5]),
        SimpleNamespace(text="hello world", tokens=[" hello", " world"], timestamps=[0.5, 1.5]),
    ]
    state = _state()
    request = SimpleNamespace(app=SimpleNamespace(state=state))
    try:
        results = await routes._redo_stalled(request, [prepared], results, "parakeet-v3:fp32")
        text, _segments, _words = await routes._stitch_request(request, prepared, results)
    finally:
        state.audio_pool.shutdown()
        state.align_pool.shutdown()
    assert text == "hello world"
    assert threads == ["audio"] and stitched == ["audio"]


@pytest.mark.asyncio
async def test_numbers_are_never_read_on_the_event_loop(calls, speak, monkeypatch):
    # Reading every number's readings back through number_parse is CPU work
    # (seconds for a long request of codes): deciding and saying both run off the loop.
    threads = []
    phrases = routes.spoken.phrases

    def recording(*args, **kwargs):
        threads.append(_pool(threading.current_thread().name))
        return phrases(*args, **kwargs)

    monkeypatch.setattr(routes.spoken, "phrases", recording)
    text = "Order 001100110011 cost £2.10 at 6pm."
    await _transcribe(response_format="json", text=text)
    await _batch(text)
    assert threads and "inline" not in threads


@pytest.mark.asyncio
async def test_a_spoken_numbers_bug_is_never_a_500(calls, speak, monkeypatch, caplog):
    def broken(_words, **_options):
        raise RuntimeError("a bug in spoken.phrases")

    monkeypatch.setattr(routes.spoken, "phrases", broken)
    body = await _transcribe(text="It cost $5 today.")
    assert body["text"] == "It cost $5 today."
    assert [w["word"] for w in body["words"]] == ["It", "cost", "$5", "today."]
    assert await _batch("It cost $5 today.") == ["It cost $5 today."]
    assert "spoken numbers failed" in caplog.text


# --- Whisper word alignment (Whisper returns text only; words come from the
# --- request's aligner, and only when the language is known) --------------------

class _WhisperWorker:
    """Whisper returns a bare transcript string per chunk: no tokens."""

    async def submit_many(self, pieces, _model_name):
        return list(pieces)


async def _transcribe_whisper(model, *, language=None, aligner_name=BASE, text="hello world"):
    state = _state()
    state.worker = _WhisperWorker()
    try:
        response = await routes.transcribe(
            request=SimpleNamespace(app=SimpleNamespace(state=state)),
            file=UploadFile(io.BytesIO(text.encode()), filename="a.wav"),
            model=model,
            quantization=None,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            timestamp_granularities_plain=None,
            language=language,
            prompt=None,
            temperature=None,
            spoken_numbers=None,
            aligner_name=aligner_name,
        )
    finally:
        state.audio_pool.shutdown()
        state.align_pool.shutdown()
    return json.loads(response.body)


@pytest.mark.asyncio
async def test_whisper_english_transcript_is_split_and_aligned(calls):
    # No tokens, so the words come purely from splitting the text and aligning.
    body = await _transcribe_whisper("whisper-base", language="en")
    assert [c["words"] for c in calls] == [WORDS]
    assert [(w["word"], w["start"], w["end"]) for w in body["words"]] == [
        ("hello", 0.1, 0.4),
        ("world", 1.6, 1.9),
    ]


@pytest.mark.asyncio
async def test_whisper_aligns_any_language_the_aligner_knows(calls):
    body = await _transcribe_whisper("whisper-base", language="de", aligner_name="omnilingual-ctc-300m")
    assert [c["language"] for c in calls] == ["de"]
    assert body["words"] is not None


@pytest.mark.asyncio
async def test_whisper_in_a_language_no_aligner_has_returns_no_words(calls):
    body = await _transcribe_whisper("whisper-base", language="ja", aligner_name=None)
    assert calls == []
    assert body["words"] is None


@pytest.mark.asyncio
async def test_whisper_multilingual_without_language_returns_no_words(calls):
    # Auto-detect could be any language; never align a multilingual model blindly.
    body = await _transcribe_whisper("whisper-base", language=None)
    assert calls == []
    assert body["words"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("why", ["the aligner failed to load", "nothing was placed"])
async def test_whisper_never_returns_placeholder_word_times(calls, monkeypatch, why):
    # Whisper has no word times: without the aligner's, it has no words at all.
    if why == "the aligner failed to load":
        monkeypatch.setattr(aligner, "for_chunk", lambda *_: None)
    else:
        monkeypatch.setattr(aligner, "for_chunk", lambda *_: SimpleNamespace(spans=lambda _words: None))
    body = await _transcribe_whisper("whisper-base.en", text="hello big wide world")
    assert body["words"] is None
    assert body["text"] == "hello big wide world"


@pytest.mark.asyncio
async def test_an_english_only_model_is_aligned_as_english(calls, monkeypatch):
    # No `language` and no default: an .en model's words are English all the same.
    monkeypatch.setattr(aligner, "ALIGN_DEFAULT_LANGUAGE", "")
    body = await _transcribe_whisper("whisper-base.en", language=None)
    assert [c["language"] for c in calls] == ["en"]
    assert [(w["word"], w["start"]) for w in body["words"]] == [("hello", 0.1), ("world", 1.6)]


@pytest.mark.asyncio
async def test_whisper_auto_language_is_not_a_known_language(calls):
    body = await _transcribe_whisper("whisper-base", language="auto")
    assert calls == []
    assert body["words"] is None


@pytest.mark.asyncio
async def test_whisper_english_only_model_aligns_without_language(calls):
    # A .en model is English by construction, so no language field is needed.
    body = await _transcribe_whisper("whisper-base.en", language=None)
    assert [c["words"] for c in calls] == [WORDS]
    assert body["words"] is not None


@pytest.mark.asyncio
async def test_whisper_words_need_an_aligner(calls):
    # Whisper has no native word times, so without an aligner there are no words.
    body = await _transcribe_whisper("whisper-base", language="en", aligner_name=None)
    assert calls == []
    assert body["words"] is None


@pytest.fixture
def retimed(calls, monkeypatch):
    """The retime_words each _stitch was given, and the pool it ran on."""
    seen = []
    stitch = routes._stitch

    def recording(*args, **kwargs):
        seen.append((kwargs.get("retime_words", False), _pool(threading.current_thread().name)))
        return stitch(*args, **kwargs)

    monkeypatch.setattr(routes, "_stitch", recording)
    return seen


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("server", "request_says", "fmt", "expected"),
    [
        (False, None, "verbose_json", False),
        (True, None, "verbose_json", True),
        (True, False, "verbose_json", False),
        (False, True, "verbose_json", True),
        (False, True, "json", False),  # no word times asked for: nothing to re-time
    ],
)
async def test_retime_words_is_the_request_s_else_the_server_s(retimed, monkeypatch, server, request_says, fmt, expected):
    monkeypatch.setattr(routes, "RETIME_WORDS", server)
    await _transcribe(response_format=fmt, retime_words=request_says)
    assert retimed == [(expected, "audio" if expected else "inline")]  # off the event loop when it works
