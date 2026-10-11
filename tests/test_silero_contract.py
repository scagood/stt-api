"""The chunker calls silero-vad's load_silero_vad and get_speech_timestamps
(chunker._load_vad, chunker._silero_speech_segments). Every other test fakes
them, and a load that fails falls back to the volume VAD with one warning, so
a silero-vad release that breaks either would ship unseen. These run the real
model on a real recording. Skipped without silero-vad and torch (CI's unit job
has neither; its onnx-asr-contract job installs them)."""

from __future__ import annotations

import threading
import wave
from pathlib import Path

import numpy as np
import pytest

from parakeet_service import chunker

pytest.importorskip("torch")
pytest.importorskip("silero_vad")
_real_ort = pytest.importorskip("onnxruntime.capi.onnxruntime_inference_collection")

SR = chunker.TARGET_SR
# LibriSpeech dev-clean 1272-135031-0009, "He doesn't work at all": 1.91 s,
# words from about 0.6 s to 1.55 s (see tests/data/README.md).
CLIP = Path(__file__).parent / "data" / "librispeech-1272-135031-0009.wav"


@pytest.fixture(autouse=True)
def _real_silero(monkeypatch):
    # conftest's autouse fixture stubs InferenceSession for the aligner guard,
    # and Silero's ONNX backend opens its session through it.
    import onnxruntime

    monkeypatch.setattr(onnxruntime, "InferenceSession", _real_ort.InferenceSession)
    # A fresh per-thread cache, so each test loads the real model itself.
    monkeypatch.setattr(chunker, "_vad_local", threading.local())


def _clip() -> np.ndarray:
    with wave.open(str(CLIP)) as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (SR, 1, 2)
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768


def test_loads_silero_not_the_volume_fallback():
    assert chunker._get_vad() != "energy"


def test_hears_the_words_and_not_the_silence_around_them():
    clip = _clip()
    pad = np.zeros(SR, dtype=np.float32)
    segments = chunker._silero_speech_segments(np.concatenate([pad, clip, pad]))

    assert segments
    assert all(type(s) is int and type(e) is int and s < e for s, e in segments)
    assert segments[0][0] >= pad.size  # nothing in the second of silence before
    assert segments[-1][1] <= pad.size + clip.size  # nor after
    # The loud part of the words, 0.75 s to 1.5 s into the clip, is all speech.
    heard = np.zeros(2 * pad.size + clip.size, dtype=bool)
    for start, end in segments:
        heard[start:end] = True
    assert heard[pad.size + int(0.75 * SR): pad.size + int(1.5 * SR)].all()


@pytest.mark.parametrize("wav", [
    np.zeros(3 * SR, dtype=np.float32),
    (np.random.default_rng(0).standard_normal(3 * SR) * 0.01).astype(np.float32),  # -40 dBFS
], ids=["silence", "noise"])
def test_hears_no_speech_without_any(wav):
    assert chunker._silero_speech_segments(wav) == []


def test_vad_filter_keeps_speech_and_drops_silence():
    assert chunker.has_speech(_clip())
    assert not chunker.has_speech(np.zeros(SR, dtype=np.float32))
