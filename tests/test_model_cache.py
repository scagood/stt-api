from __future__ import annotations

import os
import sys
import types
from collections import OrderedDict

from parakeet_service import model as m


def _stub_loader(monkeypatch, tmp_path, cache_size, calls=None):
    monkeypatch.setattr(m, "MODEL_CACHE_SIZE", cache_size)
    monkeypatch.setattr(m, "_MODELS", OrderedDict())
    monkeypatch.setattr(m, "_RUNTIMES", {})
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

    def load(model_type, folder, **_kwargs):
        if calls is not None:
            files = {name: open(os.path.join(folder, name)).read() for name in os.listdir(folder)}
            calls.append((model_type, files))
        return object()

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(m.onnx_asr, "load_model", load, raising=False)


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
    # The per-load folder of links is gone once onnx-asr has read it.
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".load-")]


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
        m.onnx_asr,
        "load_model",
        lambda *a, **k: types.SimpleNamespace(_encoder=_Session("CPUExecutionProvider")),
        raising=False,
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
        m.onnx_asr,
        "load_model",
        lambda *a, **k: types.SimpleNamespace(_encoder=_Session("CUDAExecutionProvider")),
        raising=False,
    )
    m.load_model("whisper-tiny:fp32", with_timestamps=False)
    m.load_model("whisper-base:fp32", with_timestamps=False)  # evicts tiny
    status = m.runtime_status()
    assert list(status) == ["whisper-base:fp32"]
    assert status["whisper-base:fp32"]["backend"] == "cuda"
    assert status["whisper-base:fp32"]["fallback_reason"] is None
