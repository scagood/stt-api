"""Thread-safe ONNX Runtime model loading for Parakeet TDT."""
from __future__ import annotations

import ctypes
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Tuple

# Import config before ONNX Runtime so thread-pool environment limits are active.
from .config import (
    GPU_ARENA_EXTEND_STRATEGY,
    GPU_CUDNN_ALGO_SEARCH,
    GPU_CUDNN_MAX_WORKSPACE,
    GPU_DEVICE_ID,
    GPU_MEMORY_LIMIT_MB,
    MODEL_CACHE_SIZE,
    MODEL_CONFIGS,
    MODELS_DIR,
    ORT_INTER_THREADS,
    ORT_INTRA_THREADS,
    TARGET_SR,
    USE_GPU,
    WARMUP_SEC,
    logger,
)

import numpy as np
import onnx_asr
import onnxruntime as ort

_ModelKey = Tuple[str, bool]
# Each loaded model and when it was last asked for (time.monotonic()), least
# recently used first.
_MODELS: "OrderedDict[_ModelKey, Tuple[object, float]]" = OrderedDict()
# _MODEL_LOCK guards the cache (_MODELS, _RUNTIMES, _LOADING) and is only held
# briefly: every inference (get_model) and /health take it. A load runs under
# its model's own lock in _LOADING instead: a first load downloads up to a few
# GB and builds the sessions, which should hold up only callers of that model
# (they get its result, not a second download), not hits on loaded ones.
_MODEL_LOCK = threading.RLock()
_LOADING: Dict[_ModelKey, threading.Lock] = {}
_CUDA_LOCK = threading.Lock()
_CUDA_PRELOADED = False
_RUNTIMES: Dict[str, dict] = {}


class ModelLoadError(RuntimeError):
    """A model could not be fetched or loaded. Nothing is cached: the next call retries."""


def _preload_cuda_libraries() -> bool:
    """Load CUDA/cuDNN libraries before creating any ORT session."""
    global _CUDA_PRELOADED
    if USE_GPU not in {"true", "auto"}:
        return False
    if _CUDA_PRELOADED:
        return True
    with _CUDA_LOCK:
        if _CUDA_PRELOADED:
            return True
        preload = getattr(ort, "preload_dlls", None)
        if preload is None:
            if USE_GPU == "true":
                raise RuntimeError(
                    "onnxruntime-gpu does not expose preload_dlls; install a "
                    "compatible ONNX Runtime GPU package"
                )
            return True
        try:
            # Empty directory explicitly prefers NVIDIA runtime wheels installed
            # by the [cuda,cudnn] extra over potentially incompatible system libs.
            preload(cuda=True, cudnn=True, msvc=False, directory="")
            _CUDA_PRELOADED = True
            logger.info("Preloaded CUDA/cuDNN libraries for ONNX Runtime")
            return True
        except Exception as exc:
            if USE_GPU == "true":
                raise RuntimeError("failed to preload CUDA/cuDNN libraries") from exc
            logger.warning("CUDA/cuDNN preload failed; using CPU fallback: %s", exc)
            return False


def _build_sess_options(
    intra_threads: int = ORT_INTRA_THREADS, spinning: bool = True
) -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_threads
    options.inter_op_num_threads = ORT_INTER_THREADS
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.set_denormal_as_zero", "1")
    options.add_session_config_entry(
        "session.intra_op.allow_spinning", "1" if spinning else "0"
    )
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options


def _resolve_providers() -> List[Any]:
    cuda_runtime_ready = _preload_cuda_libraries()
    available = set(ort.get_available_providers())
    has_cuda = cuda_runtime_ready and "CUDAExecutionProvider" in available

    if USE_GPU == "true" and not has_cuda:
        raise RuntimeError(
            "PARAKEET_USE_GPU=true but CUDAExecutionProvider is unavailable; "
            f"available providers: {sorted(available)}"
        )
    if USE_GPU == "false" or not has_cuda:
        return ["CPUExecutionProvider"]

    cuda_options = {
        "device_id": GPU_DEVICE_ID,
        "arena_extend_strategy": {
            "next_power_of_two": "kNextPowerOfTwo",
            "same_as_requested": "kSameAsRequested",
        }[GPU_ARENA_EXTEND_STRATEGY],
        "cudnn_conv_algo_search": GPU_CUDNN_ALGO_SEARCH.upper(),
        "cudnn_conv_use_max_workspace": "1" if GPU_CUDNN_MAX_WORKSPACE else "0",
        "do_copy_in_default_stream": "1",
    }
    if GPU_MEMORY_LIMIT_MB:
        cuda_options["gpu_mem_limit"] = GPU_MEMORY_LIMIT_MB * 1024 * 1024
    cuda = ("CUDAExecutionProvider", cuda_options)
    return [cuda] if USE_GPU == "true" else [cuda, "CPUExecutionProvider"]


def _session_provider_report(model: Any) -> Dict[str, List[str]]:
    report: Dict[str, List[str]] = {}
    candidates = [("model", model), ("asr", getattr(model, "asr", None))]
    for prefix, candidate in candidates:
        if candidate is None:
            continue
        for attribute in ("_model", "_encoder", "_decoder", "_decoder_joint"):
            session = getattr(candidate, attribute, None)
            get_providers = getattr(session, "get_providers", None)
            if callable(get_providers):
                report[f"{prefix}.{attribute}"] = list(get_providers())
    return report


def _validate_gpu_binding(name: str, model: Any) -> None:
    _check_gpu_binding(name, _session_provider_report(model))


def _check_gpu_binding(name: str, report: Dict[str, List[str]]) -> None:
    """Log the providers each of `name`'s sessions bound to; with
    PARAKEET_USE_GPU=true, raise unless every one of them is on the GPU (ONNX
    Runtime falls back to the CPU when CUDA won't start)."""
    if report:
        logger.info("Session providers for %s: %s", name, report)
    if USE_GPU != "true":
        return
    if not report:
        raise RuntimeError(
            f"PARAKEET_USE_GPU=true but ORT providers for {name} could not be inspected"
        )
    if not all(
        providers
        and providers[0] in {"CUDAExecutionProvider", "TensorrtExecutionProvider"}
        for providers in report.values()
    ):
        raise RuntimeError(
            f"PARAKEET_USE_GPU=true but {name} did not bind all sessions to GPU: {report}"
        )


def _runtime(providers: List[Any], report: Dict[str, List[str]]) -> dict:
    """What a loaded model actually runs on, for /health: the provider each
    session bound to and, when CUDA was asked for but not used, why."""
    requested_cuda = any(
        (item[0] if isinstance(item, tuple) else item) == "CUDAExecutionProvider"
        for item in providers
    )
    firsts = {names[0] for names in report.values() if names}
    if "CUDAExecutionProvider" in firsts:
        backend = "cuda"
    elif firsts:
        backend = "cpu"
    else:
        backend = "unknown"
    fallback_reason = None
    if requested_cuda and backend == "cpu":
        fallback_reason = "ONNX Runtime selected CPU instead of CUDA"
    return {"backend": backend, "sessions": report, "fallback_reason": fallback_reason}


def variant_key(model: str, quantization: str | None = None) -> str:
    """Resolve a model and optional quantization to its "model:quant" key.

    Without a quantization the answer is fp32, the reference precision, never
    something picked from the hardware: the same request gets the same numbers
    on every deployment. Raises ValueError naming the valid choices.
    """
    name = model.strip().lower()
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model {model!r}. Available models: {sorted(MODEL_CONFIGS)}")
    quant = (quantization or "fp32").strip().lower()
    available = MODEL_CONFIGS[name]["quantizations"]
    if quant not in available:
        raise ValueError(
            f"Model {name!r} has no {quant!r} quantization. Available: {list(available)}"
        )
    return f"{name}:{quant}"


def _external_tensors(graph) -> List[Any]:
    """Every initializer in `graph` and its subgraphs (If/Loop bodies) whose
    data lives in an external file."""
    import onnx

    found = [t for t in graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL]
    for node in graph.node:
        for attr in node.attribute:
            for sub in ([attr.g] if attr.HasField("g") else []) + list(attr.graphs):
                found += _external_tensors(sub)
    return found


def _model_files(variant: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    """A variant's files, by the names onnx-asr expects (models.yaml "files"),
    read straight from the HF cache, and the folder ORT reads external data
    from, if any (session.model_external_initializers_file_folder_path).

    Each value is the file's cache path, except for a model with external data
    (fp32 weights): onnxruntime refuses external data used in a subgraph when
    it is reached through a symlink, and every snapshot file in the HF cache is
    one. Such a model comes back as the bytes of its graph, with its weight
    locations pointing at the real blobs instead. Nothing is linked or copied.
    """
    from huggingface_hub import hf_hub_download

    files: Dict[str, Any] = {
        name: Path(hf_hub_download(variant["repo"], path, revision=variant["revision"]))
        for name, path in variant["files"].items()
    }
    # A model with external data has its data files listed beside it, under
    # the name the .onnx refers to (models.yaml).
    graphs = {}
    for name in files:
        if name.endswith(".onnx") and any(o != name and o.startswith(name) for o in files):
            import onnx

            proto = onnx.load(files[name], load_external_data=False)
            graphs[name] = (proto, _external_tensors(proto.graph))
    data_files = {
        kv.value
        for _, tensors in graphs.values()
        for t in tensors
        for kv in t.external_data
        if kv.key == "location"
    }
    if not data_files:
        return files, None
    blobs = {d: os.path.realpath(files[d]) for d in data_files}
    data_dir = os.path.commonpath([os.path.dirname(b) for b in blobs.values()])
    for name, (proto, tensors) in graphs.items():
        for t in tensors:
            for kv in t.external_data:
                if kv.key == "location":
                    kv.value = os.path.relpath(blobs[kv.value], data_dir)
        files[name] = proto.SerializeToString()
    return files, data_dir


def _build_asr(onnx_asr_type: str, files: Dict[str, Any], providers, sess_options):
    """onnx_asr.load_model() for files that aren't in one folder: what its
    Manager.create_asr() does, with the files handed over by role instead of
    found by globbing a directory. Leans on onnx-asr internals (the resolver's
    model types, Manager's factories); tests/test_onnx_asr_contract.py checks
    them against the installed onnx-asr.
    """
    from onnx_asr.loader import Manager
    from onnx_asr.onnx import update_onnx_providers

    model_type = _onnx_asr_model_type(onnx_asr_type)
    manager = Manager(sess_options, providers)
    config = update_onnx_providers(
        manager.default_onnx_config, excluded_providers=model_type._get_excluded_providers()
    )
    by_role = _files_by_role(model_type, files)
    return manager._create_asr_adapter(model_type(by_role, manager._create_preprocessor, config))


def _onnx_asr_model_type(onnx_asr_type: str):
    """The onnx-asr model class for a models.yaml onnx_asr_type."""
    from onnx_asr.loader import create_asr_resolver

    # The resolver only picks the class here: a type name it knows never
    # reaches the directory or the hub.
    return create_asr_resolver(onnx_asr_type, MODELS_DIR, offline=True).model_type


def _files_by_role(model_type, files: Dict[str, Any]) -> Dict[str, Any]:
    """`files` (keyed by the names onnx-asr expects) keyed by the role the
    model class reads each from, matched as its resolver globs a directory."""
    patterns = model_type._get_model_files(None)
    if "config.json" in files:
        patterns["config"] = "config.json"
    by_role = {}
    for role, pattern in patterns.items():
        matches = [n for n in files if PurePosixPath(n).full_match(pattern)]
        if len(matches) != 1:
            raise ValueError(
                f"{model_type.__name__} needs exactly one file matching {pattern!r} for "
                f"{role!r}; models.yaml gives {matches or 'none'}"
            )
        by_role[role] = files[matches[0]]
    return by_role


def load_model(key: str, *, with_timestamps: bool = True):
    """Load (or return the cached) model for a variant_key() key."""
    cache_key = (key, with_timestamps)

    with _MODEL_LOCK:
        cached = _cache_hit(cache_key)
        if cached is not None:
            return cached
        loading = _LOADING.setdefault(cache_key, threading.Lock())
    # Held through the check and the load, so a caller that waited here finds
    # the model the one before it loaded (or, after a failure, tries again).
    with loading:
        with _MODEL_LOCK:
            cached = _cache_hit(cache_key)
            if cached is not None:
                return cached
        model, runtime = _load(key, with_timestamps)
        with _MODEL_LOCK:
            _MODELS[cache_key] = (model, time.monotonic())
            _RUNTIMES[key] = runtime
            # ponytail: LRU cap, drop least-recent so a many-model sweep fits RAM.
            evicted = []
            while MODEL_CACHE_SIZE and len(_MODELS) > MODEL_CACHE_SIZE:
                evicted.append(_MODELS.popitem(last=False))
        # Their sessions go when `evicted` does: here, not under _MODEL_LOCK.
        for (evicted_key, _timestamps), _entry in evicted:
            logger.info("Evicted %s (cache size %d)", evicted_key, MODEL_CACHE_SIZE)
        logger.info("Loaded %s", key)
        return model


def _cache_hit(cache_key: _ModelKey):
    """The cached model for `cache_key`, now the most recently used, or None.
    Call under _MODEL_LOCK."""
    cached = _MODELS.pop(cache_key, None)
    if cached is None:
        return None
    _MODELS[cache_key] = (cached[0], time.monotonic())
    return cached[0]


def _load(key: str, with_timestamps: bool):
    """Download and build the model for `key`: (model, its _runtime()). Slow
    on a first download, so never call it under _MODEL_LOCK. Raises
    ModelLoadError."""
    name, _, quant = key.partition(":")
    config = MODEL_CONFIGS[name]
    variant = config["quantizations"][quant]
    try:
        providers = _resolve_providers()
        session_options = _build_sess_options()
        logger.info(
            "Loading %s from %s providers=%s intra=%d inter=%d",
            key,
            variant["repo"],
            providers,
            ORT_INTRA_THREADS,
            ORT_INTER_THREADS,
        )
        files, data_dir = _model_files(variant)
        if data_dir is not None:
            session_options = session_options or ort.SessionOptions()
            session_options.add_session_config_entry(
                "session.model_external_initializers_file_folder_path", data_dir
            )
        model = _build_asr(config["onnx_asr_type"], files, providers, session_options)
        # Verified on onnx_asr 0.12.0 (whisper-tiny): .with_timestamps() works
        # for Whisper, but the standard onnx-community/whisper-* repos carry no
        # alignment heads, so it returns empty tokens/timestamps — text only.
        # The token/timestamp shape _stitch consumes is Parakeet TDT's anyway,
        # so keep Whisper on the plain text adapter (its .recognize() returns the
        # transcript string). Whisper word times would come from forced-aligning
        # that transcript against the audio, not from the model.
        if with_timestamps and config["family"] == "parakeet":
            model = model.with_timestamps()
        _validate_gpu_binding(key, model)
    except Exception as exc:
        error = ModelLoadError(f"Model {key!r} could not be loaded: {type(exc).__name__}: {exc}")
        logger.exception("%s", error)
        raise error from exc
    return model, _runtime(providers, _session_provider_report(model))


def get_model(key: str):
    return load_model(key, with_timestamps=True)


def evict_idle(timeout: float) -> List[str]:
    """Unload the models no one has asked for in `timeout` seconds; return
    their keys. Call trim_heap() after, to give their memory back.

    Each inference asks for its model (get_model), so a model in use is never
    idle for long; one evicted mid-call lives until that call returns.
    """
    cutoff = time.monotonic() - timeout
    idle = []
    with _MODEL_LOCK:
        while _MODELS and next(iter(_MODELS.values()))[1] <= cutoff:
            idle.append(_MODELS.popitem(last=False))
    keys = [key for (key, _timestamps), _entry in idle]
    # Tear their sessions down here, not under the lock every get_model() takes.
    del idle
    for key in keys:
        logger.info("Evicted %s (unused for %.0fs)", key, timeout)
    return keys


def trim_heap() -> None:
    """Hand memory freed by unloading back to the OS. glibc keeps most of what
    ONNX Runtime frees: unloading parakeet-v3:int8 on the CPU after three 30 s
    calls took RSS from 2.0 GiB only to 1.3 GiB, and this to 0.1 GiB. A no-op
    without glibc (macOS, musl)."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def warmup_waveform(seconds: float | None = None) -> np.ndarray:
    """Build the synthetic 16 kHz waveform used for the startup warm-up pass.

    Deterministic on purpose: warm-up cost should not vary run to run. The
    tones sit in the speech band so the encoder, decoder and joint networks
    all get exercised, while the low amplitude keeps the decoder from emitting
    much text.
    """
    duration = WARMUP_SEC if seconds is None else seconds
    samples = max(1, int(duration * TARGET_SR))
    time_axis = np.arange(samples, dtype=np.float32) / TARGET_SR
    tones = sum(
        np.sin(2.0 * np.pi * frequency * time_axis) for frequency in (110.0, 220.0, 440.0)
    )
    return (0.05 * tones).astype(np.float32)


def loaded_models() -> List[str]:
    with _MODEL_LOCK:
        return sorted({key for key, _timestamps in _MODELS})


def runtime_status() -> Dict[str, dict]:
    """_runtime() for each loaded model, by variant key."""
    with _MODEL_LOCK:
        loaded = {key for key, _timestamps in _MODELS}
        return {key: _RUNTIMES[key] for key in sorted(loaded) if key in _RUNTIMES}
