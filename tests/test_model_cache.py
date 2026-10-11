from __future__ import annotations

import asyncio
import os
import sys
import threading
import types
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from parakeet_service import batchworker, config, main, routes
from parakeet_service import model as m


def _stub_loader(monkeypatch, tmp_path, cache_size, calls=None):
    monkeypatch.setattr(m, "MODEL_CACHE_SIZE", cache_size)
    monkeypatch.setattr(m, "_MODELS", OrderedDict())
    monkeypatch.setattr(m, "_RUNTIMES", {})
    monkeypatch.setattr(m, "_LOADING", {})
    monkeypatch.setattr(m, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(m, "_resolve_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(m, "_validate_gpu_binding", lambda *a, **k: None)
    monkeypatch.setattr(m, "_build_sess_options", lambda *a, **k: None)

    def download(repo, path, revision):
        # Stands in for the HF cache: one file per (repo, revision, path), naming its source.
        blob = tmp_path / "cache" / repo / revision / path
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_text(f"{repo}@{revision[:7]}/{path}")
        return str(blob)

    def build(onnx_asr_type, files, _providers, _sess_options):
        if calls is not None:
            calls.append((onnx_asr_type, {name: path.read_text() for name, path in files.items()}))
        return object()

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(m, "_build_asr", build)


def test_cache_evicts_least_recent_when_capped(monkeypatch, tmp_path):
    _stub_loader(monkeypatch, tmp_path, cache_size=2)
    a = m.load_model("whisper-tiny:fp32", with_timestamps=False)
    m.load_model("whisper-base:fp32", with_timestamps=False)
    assert m.load_model("whisper-tiny:fp32", with_timestamps=False) is a  # hit keeps it warm
    m.load_model("whisper-small:fp32", with_timestamps=False)  # evicts base (least-recent)
    assert m.loaded_models() == ["whisper-small:fp32", "whisper-tiny:fp32"]


def test_cache_unbounded_by_default(monkeypatch, tmp_path):
    _stub_loader(monkeypatch, tmp_path, cache_size=0)
    for name in ("whisper-tiny", "whisper-base", "whisper-small"):
        m.load_model(f"{name}:fp32", with_timestamps=False)
    assert len(m.loaded_models()) == 3


def test_loads_exactly_the_listed_files_under_onnx_asr_names(monkeypatch, tmp_path):
    calls = []
    _stub_loader(monkeypatch, tmp_path, cache_size=0, calls=calls)
    m.load_model("whisper-medium.en:int8", with_timestamps=False)
    repo = "Xenova/whisper-medium.en@4fbcf6e"  # the pinned revision is what's fetched
    assert calls == [
        (
            "whisper",
            {
                "encoder_model.onnx": f"{repo}/onnx/encoder_model_quantized.onnx",
                "decoder_model_merged.onnx": f"{repo}/onnx/decoder_model_merged_quantized.onnx",
                "vocab.json": f"{repo}/vocab.json",
                "added_tokens.json": f"{repo}/added_tokens.json",
                "config.json": f"{repo}/config.json",
            },
        )
    ]


def _hf_cache_with_subgraph_weights(root):
    """A model whose weights sit in an external file, inside an If subgraph (as
    in Whisper's merged decoder), laid out like the HF cache: hash-named blobs,
    reached through snapshot symlinks."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    def branch(name, k):
        w = numpy_helper.from_array(np.full((4, 4), k, dtype=np.float32), f"w{name}")
        return helper.make_graph(
            [helper.make_node("MatMul", ["x", f"w{name}"], [f"y{name}"])], name, [],
            [helper.make_tensor_value_info(f"y{name}", TensorProto.FLOAT, [1, 4])], [w],
        )

    graph = helper.make_graph(
        [helper.make_node("If", ["c"], ["y"], then_branch=branch("t", 1.0), else_branch=branch("e", 2.0))],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4]),
         helper.make_tensor_value_info("c", TensorProto.BOOL, [])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
    )
    build = root / "build"
    build.mkdir(parents=True)
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10),
        build / "decoder.onnx", save_as_external_data=True, location="decoder.onnx_data", size_threshold=0,
    )
    blobs, snapshot = root / "blobs", root / "snapshots" / "rev"
    blobs.mkdir()
    snapshot.mkdir(parents=True)
    for name, blob in (("decoder.onnx", "aaa"), ("decoder.onnx_data", "bbb")):
        (build / name).rename(blobs / blob)
        (snapshot / name).symlink_to(os.path.relpath(blobs / blob, snapshot))
    return snapshot


def test_loads_subgraph_weights_from_the_hf_cache_without_links(monkeypatch, tmp_path):
    # onnxruntime refuses a symlink to external data used in a subgraph, so the
    # graph handed to it must point at the real blob, not a link to it. Needs
    # the real onnx and onnxruntime (CI's unit job stubs onnxruntime; the
    # autouse fixture stubs InferenceSession).
    import numpy as np

    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime.capi.onnxruntime_inference_collection")

    snapshot = _hf_cache_with_subgraph_weights(tmp_path / "cache")
    download = lambda repo, path, revision: str(snapshot / path)  # noqa: E731
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=download))
    variant = {"repo": "r", "revision": "rev", "files": {n: n for n in ("decoder.onnx", "decoder.onnx_data")}}

    files, data_dir = m._model_files(variant)
    options = pytest.importorskip("onnxruntime").SessionOptions()
    options.add_session_config_entry("session.model_external_initializers_file_folder_path", data_dir)
    session = ort.InferenceSession(files["decoder.onnx"], options, providers=["CPUExecutionProvider"])
    out = session.run(None, {"x": np.ones((1, 4), np.float32), "c": np.array(True)})[0]
    assert out.tolist() == [[4.0, 4.0, 4.0, 4.0]]

class _Session:
    def __init__(self, provider):
        self.provider = provider

    def get_providers(self):
        return [self.provider, "CPUExecutionProvider"]


def test_runtime_reports_a_cpu_fallback_when_cuda_was_asked_for(monkeypatch, tmp_path):
    _stub_loader(monkeypatch, tmp_path, cache_size=0)
    cuda = ("CUDAExecutionProvider", {})
    monkeypatch.setattr(m, "_resolve_providers", lambda: [cuda, "CPUExecutionProvider"])
    monkeypatch.setattr(
        m, "_build_asr", lambda *a, **k: types.SimpleNamespace(_encoder=_Session("CPUExecutionProvider"))
    )
    m.load_model("whisper-tiny:fp32", with_timestamps=False)
    assert m.runtime_status() == {
        "whisper-tiny:fp32": {
            "backend": "cpu",
            "sessions": {"model._encoder": ["CPUExecutionProvider", "CPUExecutionProvider"]},
            "fallback_reason": "ONNX Runtime selected CPU instead of CUDA",
        }
    }


def test_runtime_reports_cuda_and_forgets_evicted_models(monkeypatch, tmp_path):
    _stub_loader(monkeypatch, tmp_path, cache_size=1)
    monkeypatch.setattr(m, "_resolve_providers", lambda: [("CUDAExecutionProvider", {})])
    monkeypatch.setattr(
        m, "_build_asr", lambda *a, **k: types.SimpleNamespace(_encoder=_Session("CUDAExecutionProvider"))
    )
    m.load_model("whisper-tiny:fp32", with_timestamps=False)
    m.load_model("whisper-base:fp32", with_timestamps=False)  # evicts tiny
    status = m.runtime_status()
    assert list(status) == ["whisper-base:fp32"]
    assert status["whisper-base:fp32"]["backend"] == "cuda"
    assert status["whisper-base:fp32"]["fallback_reason"] is None


def test_models_unused_for_the_idle_timeout_are_evicted(monkeypatch, tmp_path):
    _stub_loader(monkeypatch, tmp_path, cache_size=0)
    now = [1000.0]
    monkeypatch.setattr(m.time, "monotonic", lambda: now[0])
    tiny = m.load_model("whisper-tiny:fp32", with_timestamps=False)
    m.load_model("whisper-base:fp32", with_timestamps=False)
    now[0] += 50
    assert m.load_model("whisper-tiny:fp32", with_timestamps=False) is tiny  # a hit is a use
    now[0] += 60
    assert m.evict_idle(100) == ["whisper-base:fp32"]
    assert m.loaded_models() == ["whisper-tiny:fp32"]
    assert m.evict_idle(100) == []
    now[0] += 50
    assert m.evict_idle(100) == ["whisper-tiny:fp32"]
    assert m.loaded_models() == [] and m.runtime_status() == {}
    assert m.load_model("whisper-tiny:fp32", with_timestamps=False) is not tiny  # loaded again on use


def test_idle_models_are_torn_down_outside_the_model_lock(monkeypatch, tmp_path):
    _stub_loader(monkeypatch, tmp_path, cache_size=0)
    held = []

    class Model:
        pass

    def load(*_a, **_k):
        made = Model()
        weakref.finalize(made, lambda: held.append(m._MODEL_LOCK._is_owned()))
        return made

    monkeypatch.setattr(m, "_build_asr", load)
    m.load_model("whisper-tiny:fp32", with_timestamps=False)
    assert m.evict_idle(0) == ["whisper-tiny:fp32"]
    assert held == [False]  # gone, and not while every get_model() waited on it


def _held_build(monkeypatch, key):
    """Make building `key`'s model set `started`, then wait for `release`:
    (started, release, builds), builds counting every model built."""
    started, release, builds = threading.Event(), threading.Event(), []
    repo = config.MODEL_CONFIGS[key.partition(":")[0]]["quantizations"][key.partition(":")[2]]["repo"]
    build = m._build_asr

    def held(onnx_asr_type, files, providers, sess_options):
        builds.append(onnx_asr_type)
        if any(repo in path.read_text() for path in files.values()):
            started.set()
            assert release.wait(5)
        return build(onnx_asr_type, files, providers, sess_options)

    monkeypatch.setattr(m, "_build_asr", held)
    return started, release, builds


def test_a_first_load_does_not_hold_up_loaded_models_or_health(monkeypatch, tmp_path):
    _stub_loader(monkeypatch, tmp_path, cache_size=0)
    tiny = m.load_model("whisper-tiny:fp32", with_timestamps=False)
    started, release, _builds = _held_build(monkeypatch, "whisper-base:fp32")
    request = types.SimpleNamespace(app=types.SimpleNamespace(state=types.SimpleNamespace(ready=True)))
    with ThreadPoolExecutor(2) as pool:
        try:
            first = pool.submit(m.load_model, "whisper-base:fp32", with_timestamps=False)
            assert started.wait(5)
            # While base is still loading:
            hit = pool.submit(m.load_model, "whisper-tiny:fp32", with_timestamps=False)
            assert hit.result(timeout=2) is tiny
            health = pool.submit(routes.health, request).result(timeout=2)
            assert health["loaded"] == ["whisper-tiny:fp32"] and list(health["runtime"]) == ["whisper-tiny:fp32"]
        finally:
            release.set()
        first.result(timeout=5)
    assert m.loaded_models() == ["whisper-base:fp32", "whisper-tiny:fp32"]


def test_a_failed_load_does_not_hold_up_loaded_models_and_is_retried(monkeypatch, tmp_path):
    _stub_loader(monkeypatch, tmp_path, cache_size=0)
    tiny = m.load_model("whisper-tiny:fp32", with_timestamps=False)
    started, release, builds = _held_build(monkeypatch, "whisper-base:fp32")
    held = m._build_asr
    fail = [True]

    def failing(*args):
        model = held(*args)
        if fail[0]:
            raise OSError("no network")
        return model

    monkeypatch.setattr(m, "_build_asr", failing)
    with ThreadPoolExecutor(2) as pool:
        try:
            first = pool.submit(m.load_model, "whisper-base:fp32", with_timestamps=False)
            assert started.wait(5)
            assert pool.submit(m.load_model, "whisper-tiny:fp32", with_timestamps=False).result(timeout=2) is tiny
        finally:
            release.set()
        with pytest.raises(m.ModelLoadError, match="no network"):
            first.result(timeout=5)
    assert m.loaded_models() == ["whisper-tiny:fp32"]
    fail[0] = False
    m.load_model("whisper-base:fp32", with_timestamps=False)  # nothing cached: tried again
    assert m.loaded_models() == ["whisper-base:fp32", "whisper-tiny:fp32"]
    assert builds == ["whisper", "whisper"]


def test_concurrent_first_loads_of_a_model_build_it_once(monkeypatch, tmp_path):
    _stub_loader(monkeypatch, tmp_path, cache_size=0)
    started, release, builds = _held_build(monkeypatch, "whisper-base:fp32")
    reached = threading.Semaphore(0)

    class Reached:
        """The model's load lock, counting the callers that reach it."""

        def __init__(self):
            self._lock = threading.Lock()

        def __enter__(self):
            reached.release()
            return self._lock.__enter__()

        def __exit__(self, *exc):
            return self._lock.__exit__(*exc)

    monkeypatch.setitem(m._LOADING, ("whisper-base:fp32", False), Reached())
    with ThreadPoolExecutor(2) as pool:
        try:
            loads = [pool.submit(m.load_model, "whisper-base:fp32", with_timestamps=False) for _ in range(2)]
            # Both reach it before the first build ends: the second waits for it.
            assert reached.acquire(timeout=5) and reached.acquire(timeout=5)
        finally:
            release.set()
        first, second = (load.result(timeout=5) for load in loads)
    assert first is second
    assert builds == ["whisper"]


class _Recognizer:
    def recognize(self, wav):
        return [len(w) for w in wav] if isinstance(wav, list) else len(wav)


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_type", [batchworker.InferencePool, batchworker.BatchWorker])
async def test_requests_waiting_on_a_first_load_do_not_hold_up_loaded_models(monkeypatch, tmp_path, worker_type):
    _stub_loader(monkeypatch, tmp_path, cache_size=0)
    monkeypatch.setattr(m, "_build_asr", lambda *a, **k: _Recognizer())
    m.load_model("whisper-tiny:fp32")
    started, release, _builds = _held_build(monkeypatch, "whisper-base:fp32")
    loop = asyncio.get_running_loop()
    default = ThreadPoolExecutor(2)  # as min(32, cpus + 4) is, once that many wait
    loop.set_default_executor(default)
    worker = worker_type(m.get_model, cached_fn=m.cached_model)
    await worker.start()
    wav = np.zeros(160, np.float32)
    try:
        cold = [asyncio.create_task(worker.submit_many([wav], "whisper-base:fp32")) for _ in range(2)]
        assert await asyncio.to_thread(started.wait, 5)
        # Both threads of the default executor now wait on base's load.
        assert await asyncio.wait_for(worker.submit_many([wav], "whisper-tiny:fp32"), 2) == [160]
    finally:
        release.set()
    assert await asyncio.wait_for(asyncio.gather(*cold), 5) == [[160], [160]]
    await worker.stop()
    default.shutdown()


def test_preloading_cuda_does_not_take_the_model_lock(monkeypatch):
    held = []
    monkeypatch.setattr(m, "USE_GPU", "auto")
    monkeypatch.setattr(m, "_CUDA_PRELOADED", False)
    monkeypatch.setattr(
        m.ort, "preload_dlls", lambda **_k: held.append(m._MODEL_LOCK._is_owned()), raising=False
    )
    assert m._preload_cuda_libraries()
    assert held == [False]


def test_trim_heap_never_raises():
    m.trim_heap()  # glibc's malloc_trim here; a no-op where there is none


@pytest.mark.parametrize(("raw", "expected"), [(None, 21600.0), ("-1", -1.0), ("0.5", 0.5), ("3600", 3600.0)])
def test_idle_timeout_setting_parses(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("PARAKEET_MODEL_IDLE_TIMEOUT_SEC", raising=False)
    else:
        monkeypatch.setenv("PARAKEET_MODEL_IDLE_TIMEOUT_SEC", raw)
    assert config._env_timeout("PARAKEET_MODEL_IDLE_TIMEOUT_SEC", 21600.0) == expected


@pytest.mark.parametrize("raw", ["0", "-0.5", "-2", "6h"])
def test_idle_timeout_setting_rejects_zero_and_other_negatives(monkeypatch, raw):
    monkeypatch.setenv("PARAKEET_MODEL_IDLE_TIMEOUT_SEC", raw)
    with pytest.raises(RuntimeError, match="PARAKEET_MODEL_IDLE_TIMEOUT_SEC"):
        config._env_timeout("PARAKEET_MODEL_IDLE_TIMEOUT_SEC", 21600.0)


def test_unloading_trims_the_heap_only_when_something_was_unloaded(monkeypatch):
    trims, unloaded = [], {"models": [], "aligners": []}
    monkeypatch.setattr(main, "trim_heap", lambda: trims.append(True))
    monkeypatch.setattr(main, "evict_idle", lambda timeout: unloaded["models"])
    monkeypatch.setattr(main.aligner, "evict_idle", lambda timeout: unloaded["aligners"])
    main._unload_idle(10)
    assert trims == []
    unloaded["aligners"] = ["wav2vec2-base-960h:int8"]
    main._unload_idle(10)
    assert trims == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize(("timeout", "unloads"), [(0.01, True), (-1, False)])
async def test_lifespan_unloads_idle_models_until_shutdown(monkeypatch, timeout, unloads):
    from fastapi import FastAPI

    calls = []
    monkeypatch.setattr(main, "PRELOAD_MODELS", [])
    monkeypatch.setattr(main, "MODEL_IDLE_TIMEOUT_SEC", timeout)
    monkeypatch.setattr(main, "_unload_idle", calls.append)
    async with main.lifespan(FastAPI()):
        await asyncio.sleep(0.1)
    assert bool(calls) is unloads and set(calls) <= {timeout}
    stopped = len(calls)
    await asyncio.sleep(0.05)
    assert len(calls) == stopped  # nothing after shutdown
