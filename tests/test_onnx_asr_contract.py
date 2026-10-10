"""stt-api builds onnx-asr models from files in the HF cache without the folder
onnx_asr.load_model() wants (model._build_asr), which leans on onnx-asr
internals. These run against the installed onnx-asr and onnxruntime, so an
onnx-asr upgrade that moves them fails here. Skipped without them (CI's unit
job stubs both; its onnx-asr job installs them)."""

from __future__ import annotations

import json

import pytest

from parakeet_service import model as m
from parakeet_service.config import MODEL_CONFIGS

pytest.importorskip("onnx_asr.loader")
pytest.importorskip("onnx")
_real_ort = pytest.importorskip("onnxruntime.capi.onnxruntime_inference_collection")


@pytest.fixture(autouse=True)
def _real_inference_session(monkeypatch):
    # conftest's autouse fixture stubs InferenceSession for the aligner guard.
    import onnxruntime

    monkeypatch.setattr(onnxruntime, "InferenceSession", _real_ort.InferenceSession)


VARIANTS = [
    pytest.param(config["onnx_asr_type"], variant["files"], id=f"{name}:{quant}")
    for name, config in MODEL_CONFIGS.items()
    for quant, variant in config["quantizations"].items()
]


@pytest.mark.parametrize("onnx_asr_type, files", VARIANTS)
def test_every_catalog_variant_fills_every_role_its_model_reads(onnx_asr_type, files):
    model_type = m._onnx_asr_model_type(onnx_asr_type)
    by_role = m._files_by_role(model_type, {name: name for name in files})
    assert set(by_role) >= set(model_type._get_model_files(None))


def _graph(path, inputs, outputs):
    """A model with the given input and output names, each a float tensor of
    unknown shape: enough for onnx-asr to open a session on."""
    import onnx
    from onnx import TensorProto, helper

    nodes = [helper.make_node("Identity", [inputs[0]], [out]) for out in outputs]
    graph = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info(i, TensorProto.FLOAT, None) for i in inputs],
        [helper.make_tensor_value_info(o, TensorProto.FLOAT, None) for o in outputs],
    )
    onnx.save_model(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10), path)


def _fake_files(tmp_path, onnx_asr_type, names):
    files = {}
    for name in names:
        path = tmp_path / name
        if name.endswith(".onnx"):
            _graph(path, ["x"], ["y"])
        elif name == "vocab.txt":
            path.write_text("<blk> 0\n▁a 1\n")
        elif name == "vocab.json":
            path.write_text(json.dumps({"a": 0}))
        elif name == "added_tokens.json":
            special = ("<|endoftext|>", "<|startoftranscript|>", "<|transcribe|>", "<|notimestamps|>")
            path.write_text(json.dumps({token: i + 1 for i, token in enumerate(special)}))
        elif name.endswith(".json"):
            path.write_text("{}")
        else:
            continue  # external data: not handed to onnx-asr
        files[name] = path
    return files


@pytest.mark.parametrize("onnx_asr_type", sorted({c["onnx_asr_type"] for c in MODEL_CONFIGS.values()}))
def test_builds_each_model_type_from_files_outside_one_folder(tmp_path, onnx_asr_type):
    names = next(
        v["files"] for c in MODEL_CONFIGS.values() if c["onnx_asr_type"] == onnx_asr_type
        for v in c["quantizations"].values()
    )
    files = _fake_files(tmp_path, onnx_asr_type, names)
    asr = m._build_asr(onnx_asr_type, files, ["CPUExecutionProvider"], None)
    assert hasattr(asr, "recognize") and hasattr(asr, "with_timestamps")
