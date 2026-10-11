"""GET / and /transcribe: the transcription page, served only with PARAKEET_TRANSCRIBE_UI on."""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from parakeet_service import routes


def _request(docs_url="/docs"):
    return SimpleNamespace(app=SimpleNamespace(docs_url=docs_url))


def _page(monkeypatch, docs_url="/docs") -> str:
    monkeypatch.setattr(routes, "TRANSCRIBE_UI", True)
    page = routes.transcribe_page(_request(docs_url))
    assert page.media_type == "text/html"
    return page.body.decode("utf-8")


def _settings(html: str) -> dict:
    found = re.search(r'<script id="server" type="application/json">(.*?)</script>', html)
    return json.loads(found.group(1))


def test_transcribe_page_is_a_404_when_off(monkeypatch):
    monkeypatch.setattr(routes, "TRANSCRIBE_UI", False)
    with pytest.raises(HTTPException) as caught:
        routes.transcribe_page(_request())
    assert caught.value.status_code == 404


def test_transcribe_page_does_not_follow_the_compare_flag(monkeypatch):
    monkeypatch.setattr(routes, "COMPARE_UI", True)
    monkeypatch.setattr(routes, "TRANSCRIBE_UI", False)
    with pytest.raises(HTTPException):
        routes.transcribe_page(_request())


def test_transcribe_page_calls_only_routes_that_exist(monkeypatch):
    urls = set(re.findall(r'"(v1/[\w/]+)"', _page(monkeypatch)))
    assert urls == {"v1/models", "v1/aligners", "v1/audio/transcriptions"}
    assert {"/" + url for url in urls} <= {route.path for route in routes.router.routes}


def test_transcribe_page_sends_only_fields_the_endpoint_takes(monkeypatch):
    sent = set(re.findall(r'form\.append\("([^"]+)"', _page(monkeypatch)))
    endpoint = next(
        route for route in routes.router.routes
        if route.path == "/v1/audio/transcriptions" and "POST" in route.methods
    )
    taken = {field.alias for field in endpoint.dependant.body_params}
    assert sent <= taken
    assert {"file", "model", "response_format", "timestamp_granularities[]"} <= sent


def test_transcribe_page_is_given_the_server_limits_and_defaults(monkeypatch):
    monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 12345)
    monkeypatch.setattr(routes, "MAX_AUDIO_SECONDS", 678.0)
    monkeypatch.setattr(routes, "SPOKEN_NUMBERS", True)
    monkeypatch.setattr(routes, "VAD_FILTER", False)
    monkeypatch.setattr(routes, "RETIME_WORDS", True)
    monkeypatch.setattr(routes, "COMPARE_UI", False)
    monkeypatch.setattr(routes, "PRELOAD_MODELS", [])
    html = _page(monkeypatch)
    assert _settings(html) == {
        "model": None,
        "max_upload_bytes": 12345,
        "max_audio_seconds": 678.0,
        "spoken_numbers": True,
        "vad_filter": False,
        "retime_words": True,
        "links": {"docs": "docs", "compare": None},
    }
    assert "__SERVER_SETTINGS__" not in html


def test_transcribe_page_links_only_the_pages_this_server_shows(monkeypatch):
    monkeypatch.setattr(routes, "COMPARE_UI", True)
    assert _settings(_page(monkeypatch))["links"] == {"docs": "docs", "compare": "compare"}
    monkeypatch.setattr(routes, "COMPARE_UI", False)
    assert _settings(_page(monkeypatch, docs_url=None))["links"] == {"docs": None, "compare": None}


def test_transcribe_page_is_served_at_the_root_too():
    paths = {route.path for route in routes.router.routes if getattr(route, "endpoint", None) is routes.transcribe_page}
    assert paths == {"/", "/transcribe"}


def test_transcribe_page_starts_on_the_first_preloaded_model(monkeypatch):
    model = next(iter(routes.MODEL_CONFIGS))
    quantization = list(routes.MODEL_CONFIGS[model]["quantizations"])[-1]
    monkeypatch.setattr(routes, "PRELOAD_MODELS", [f"{model}:{quantization}", model])
    assert _settings(_page(monkeypatch))["model"] == f"{model}:{quantization}"
    # Without a quantization a preloaded model is fp32, as the server loads it.
    monkeypatch.setattr(routes, "PRELOAD_MODELS", [model])
    assert _settings(_page(monkeypatch))["model"] == f"{model}:fp32"


def test_the_root_is_a_404_when_the_page_is_off(monkeypatch):
    """With PARAKEET_TRANSCRIBE_UI off, / is as it was before the page: a 404."""
    monkeypatch.setattr(routes, "TRANSCRIBE_UI", False)
    root = next(route for route in routes.router.routes if route.path == "/")
    with pytest.raises(HTTPException) as caught:
        root.endpoint(_request())
    assert caught.value.status_code == 404


def test_transcribe_page_tells_whisper_models_by_their_catalog_family(monkeypatch):
    """The page warns that a Whisper model gives word times only with an aligner.
    It finds Whisper models by /v1/models' owned_by, which follows the catalog
    family, not by the model's name."""
    html = _page(monkeypatch)
    assert 'owned_by === "openai"' in html
    assert "/^whisper/" not in html
    names = {name: routes._family(name) for name in routes.MODEL_CONFIGS}
    assert "whisper" in names.values()
    listed = {model["id"]: model["owned_by"] for model in routes.list_models()["data"]}
    for name, family in names.items():
        assert (listed[name] == "openai") == (family == "whisper")
