"""Fake-decode harness: run the same scenario against a given worktree's routes.

Usage: import with the worktree root first on sys.path.
"""
from __future__ import annotations

import asyncio
import math
import sys
import types
from types import SimpleNamespace

import numpy as np

try:
    import onnx_asr  # noqa: F401
    import onnxruntime  # noqa: F401
except ImportError:
    ort_stub = types.ModuleType("onnxruntime")
    ort_stub.get_available_providers = lambda: ["CPUExecutionProvider"]
    sys.modules.setdefault("onnx_asr", types.ModuleType("onnx_asr"))
    sys.modules.setdefault("onnxruntime", ort_stub)

from parakeet_service import routes  # noqa: E402
from parakeet_service.config import TARGET_SR  # noqa: E402

SR = TARGET_SR
FRAME = 0.08


def S(x):
    return int(round(x * SR))


class Model:
    """A fake Parakeet: `truth` is [(tokens, [abs token times])] per word.
    `skips(a, b)` -> list of (lo, hi) abs seconds it skips for window a..b.
    `bias(a, b)` -> seconds added to every time in that decode (a frame early, say)."""

    def __init__(self, truth, skips=lambda a, b: [], bias=lambda a, b: 0.0, made_up=lambda a, b: []):
        self.truth, self.skips, self.bias, self.made_up = truth, skips, bias, made_up
        self.calls = []

    def decode(self, a, b):
        self.calls.append((round(a, 3), round(b, 3)))
        skipped = self.skips(a, b)
        bias = self.bias(a, b)
        tokens, times = [], []
        words = list(self.truth) + list(self.made_up(a, b))
        words.sort(key=lambda w: w[1][0])
        for toks, ts in words:
            t0 = ts[0]
            if not (a <= t0 < b):
                continue
            if any(lo <= t0 < hi for lo, hi in skipped):
                continue
            for tok, t in zip(toks, ts):
                rel = max(0.0, round((t - a) / FRAME) * FRAME + bias)
                tokens.append(tok)
                times.append(round(rel, 4))
        text = "".join(t.replace("▁", " ") for t in tokens).strip()
        return SimpleNamespace(text=text, tokens=tokens, timestamps=times)


class Worker:
    def __init__(self, model):
        self.model = model

    async def submit_many(self, pieces, _key):
        out = []
        for piece in pieces:
            a = float(piece[0]) / SR
            b = a + piece.size / SR
            out.append(self.model.decode(a, b))
        return out


def request(worker):
    state = SimpleNamespace(worker=worker, ready=True, audio_pool=None, align_pool=None)
    return SimpleNamespace(app=SimpleNamespace(state=state))


def prepared(total, ranges, windows, speech):
    wav = np.arange(S(total), dtype=np.float64)
    r = [(S(a), S(b)) for a, b in ranges]
    w = [(S(a), S(b)) for a, b in windows]
    sp = None if speech is None else [(S(a), S(b)) for a, b in speech]
    return routes._PreparedAudio(
        waveform=wav, ranges=r, windows=w, speech=sp,
        pieces=[wav[a:b] for a, b in w], duration=total,
    )


def run(total, ranges, windows, speech, model, vad=None, model_key="parakeet-v3:fp32", **stitch):
    if vad is not None and hasattr(routes, "speech_segments"):
        routes.speech_segments = lambda wav: [(S(a), S(b)) for a, b in vad]
    prep = prepared(total, ranges, windows, speech)
    worker = Worker(model)
    req = request(worker)

    async def go():
        results = await worker.submit_many(prep.pieces, model_key)
        model.calls_first = list(model.calls)
        results = await routes._redo_stalled(req, [prep], results, model_key)
        return results

    results = asyncio.run(go())
    text, segments, words = routes._stitch(prep, results, **stitch)
    return SimpleNamespace(text=text, segments=segments, words=words, calls=model.calls, prep=prep, results=results)


def words_every(start, stop, step, prefix="w", tokens_per_word=1, token_step=0.16):
    out = []
    t, i = start, 0
    while t < stop - 1e-9:
        if tokens_per_word == 1:
            out.append(([f" {prefix}{i}"], [round(t, 4)]))
        else:
            toks = [f" {prefix}{i}"] + [f"_{k}" for k in range(1, tokens_per_word)]
            out.append((toks, [round(t + k * token_step, 4) for k in range(tokens_per_word)]))
        t += step
        i += 1
    return out


def word_list(truth):
    return [("".join(toks).replace("▁", " ").strip(), ts[0]) for toks, ts in truth]
