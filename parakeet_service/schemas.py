"""Response shapes, for the OpenAPI docs only.

The routes build their responses as plain dicts and never validate them
against these; tests/test_schemas.py checks that what they return matches.
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class _Schema(BaseModel):
    # A key the route adds but the schema lacks fails the tests, not a request.
    model_config = ConfigDict(extra="forbid")


class ErrorResponse(_Schema):
    detail: str = Field(description="What was wrong with the request.")


class ModelCard(_Schema):
    """A model a request may name in `model`."""

    id: str
    object: Literal["model"]
    created: int = Field(description="Unix time; fixed, not when it was loaded.")
    owned_by: str = Field(description="The model's author.")
    language: List[str] = Field(description="ISO 639-1 codes it transcribes.")
    quantizations: List[str] = Field(description="Precisions it comes in; `fp32` is the default.")
    task: Literal["automatic-speech-recognition"]


class ModelList(_Schema):
    object: Literal["list"]
    data: List[ModelCard]


class AlignerCard(_Schema):
    """A forced aligner a request may name in `aligner`."""

    id: str
    object: Literal["aligner"]
    created: int = Field(description="Unix time; fixed, not when it was loaded.")
    language: List[str] = Field(description="ISO 639-1 codes it aligns.")
    quantizations: List[str]
    default_quantization: str
    task: Literal["forced-alignment"]


class AlignerList(_Schema):
    object: Literal["list"]
    data: List[AlignerCard]


class ModelRuntime(_Schema):
    """What a loaded model runs on."""

    backend: Literal["cuda", "cpu", "unknown"]
    sessions: Dict[str, List[str]] = Field(
        description="Each ONNX Runtime session's execution providers, in order."
    )
    fallback_reason: Optional[str] = Field(description="Why it is on the CPU when CUDA was asked for.")


class CpuInfo(_Schema):
    """CPU counts, and the thread pools sized from them."""

    physical: int
    logical: int
    detected_physical: int
    detected_logical: int
    cgroup_quota: Optional[int] = Field(description="CPUs the cgroup's CFS quota allows, if one is set.")
    ort_intra: int
    ort_inter: int
    audio_workers: int
    infer_workers: int
    align_threads: int


class Health(_Schema):
    status: Literal["healthy", "starting"]
    ready: bool
    models: List[str] = Field(description="Every model in the catalog.")
    loaded: List[str] = Field(description="Loaded models, as `model:quantization`.")
    runtime: Dict[str, ModelRuntime] = Field(description="By `model:quantization`.")
    cpu: CpuInfo
    aligner: Dict[str, Literal["loaded", "failed", "not loaded"]] = Field(
        description="Each aligner, by `aligner:quantization`."
    )


class Ready(_Schema):
    status: Literal["ok"]


class Transcription(_Schema):
    """`response_format=json`."""

    text: str


class TranscriptionWord(_Schema):
    start: float
    end: float
    word: str


class TranscriptionSegment(_Schema):
    """`tokens`, `temperature`, `avg_logprob`, `compression_ratio` and
    `no_speech_prob` are always empty or zero, for OpenAI clients that expect them."""

    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class VerboseTranscription(_Schema):
    """`response_format=verbose_json`."""

    task: Literal["transcribe"]
    language: str = Field(description="The request's `language`, or `auto`.")
    duration: float = Field(description="Seconds of audio.")
    text: str
    segments: List[TranscriptionSegment]
    words: Optional[List[TranscriptionWord]] = Field(
        description="With `timestamp_granularities[]=word`, else null."
    )


class BatchItem(_Schema):
    filename: str
    text: str
    duration: float = Field(description="Seconds of audio.")


class BatchTranscription(_Schema):
    results: List[BatchItem] = Field(description="One per file, in the order sent.")
    batch_size: int
