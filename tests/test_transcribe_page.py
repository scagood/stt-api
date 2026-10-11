"""GET /transcribe: the transcription page, served only with PARAKEET_TRANSCRIBE_UI on."""
from __future__ import annotations

import json
import re

import pytest
from fastapi import HTTPException

from parakeet_service import routes


def _page(monkeypatch) -> str:
    monkeypatch.setattr(routes, "TRANSCRIBE_UI", True)
    page = routes.transcribe_page()
    assert page.media_type == "text/html"
    return page.body.decode("utf-8")


def test_transcribe_page_is_a_404_when_off(monkeypatch):
    monkeypatch.setattr(routes, "TRANSCRIBE_UI", False)
    with pytest.raises(HTTPException) as caught:
        routes.transcribe_page()
    assert caught.value.status_code == 404


def test_transcribe_page_does_not_follow_the_compare_flag(monkeypatch):
    monkeypatch.setattr(routes, "COMPARE_UI", True)
    monkeypatch.setattr(routes, "TRANSCRIBE_UI", False)
    with pytest.raises(HTTPException):
        routes.transcribe_page()


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
    html = _page(monkeypatch)
    settings = re.search(r'<script id="server" type="application/json">(.*?)</script>', html).group(1)
    assert json.loads(settings) == {
        "max_upload_bytes": 12345,
        "max_audio_seconds": 678.0,
        "spoken_numbers": True,
        "vad_filter": False,
        "retime_words": True,
    }
    assert "__SERVER_SETTINGS__" not in html
