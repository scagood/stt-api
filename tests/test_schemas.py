"""The OpenAPI docs: the response schemas in parakeet_service.schemas match
what the routes return, and the docs' examples match the schemas.

The routes never validate their responses against the schemas; these tests
are what keeps the two in step. Transcription responses are checked in
test_routes_words.py, which fakes the model.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from pydantic import TypeAdapter

from parakeet_service import model, routes, schemas


def _openapi():
    app = FastAPI()
    app.include_router(routes.router)
    return app.openapi()


def _request(ready):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(ready=ready)))


def test_model_and_aligner_cards_match_their_schemas():
    schemas.ModelList.model_validate(routes.list_models())
    schemas.AlignerList.model_validate(routes.list_aligners())
    for name in routes.MODEL_CONFIGS:
        schemas.ModelCard.model_validate(routes.retrieve_model(name))
    for name in routes.ALIGNER_CONFIGS:
        schemas.AlignerCard.model_validate(routes.retrieve_aligner(name))


@pytest.mark.parametrize("ready", [True, False])
def test_health_matches_its_schema(ready):
    schemas.Health.model_validate(routes.health(_request(ready)))


def test_a_loaded_model_s_runtime_matches_its_schema():
    report = {"asr._encoder": ["CPUExecutionProvider"]}
    schemas.ModelRuntime.model_validate(model._runtime(["CUDAExecutionProvider"], report))


def test_healthz_matches_its_schemas():
    schemas.Ready.model_validate(routes.healthz(_request(True)))
    with pytest.raises(HTTPException) as raised:
        routes.healthz(_request(False))
    schemas.ErrorResponse.model_validate({"detail": raised.value.detail})


def _documented_examples():
    for route in routes.router.routes:
        for code, response in getattr(route, "responses", {}).items():
            content = response.get("content", {}).get("application/json", {})
            examples = [content["example"]] if "example" in content else []
            examples += [example["value"] for example in content.get("examples", {}).values()]
            for example in examples:
                yield pytest.param(response["model"], example, id=f"{route.name}-{code}")


@pytest.mark.parametrize(("schema", "example"), list(_documented_examples()))
def test_docs_examples_match_their_schemas(schema, example):
    TypeAdapter(schema).validate_python(example)


def test_every_route_documents_its_response():
    for path, operations in _openapi()["paths"].items():
        for method, operation in operations.items():
            assert operation["responses"]["200"]["content"]["application/json"]["schema"], (method, path)


def test_form_fields_are_described_and_never_null():
    # Swagger UI shows an optional field's null as `string | (string | null)`;
    # a form leaves the field out instead.
    components = _openapi()["components"]["schemas"]
    for body in ("Body_transcribe", "Body_transcribe_batch"):
        for name, field in components[body]["properties"].items():
            assert "anyOf" not in field, (body, name)
            assert field.get("description"), (body, name)
