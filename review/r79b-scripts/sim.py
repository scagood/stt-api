"""Simulate Parakeet over truth words for a given worktree's routes.

Usage: python sim.py <worktree> <scenario> [args]
Imports parakeet_service from <worktree> (with onnx stubs)."""
from __future__ import annotations

import asyncio
import math
import os
import random
import sys
import types
from types import SimpleNamespace

WT = sys.argv[1]
sys.path.insert(0, WT)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ort_stub = types.ModuleType("onnxruntime")
ort_stub.get_available_providers = lambda: ["CPUExecutionProvider"]
sys.modules.setdefault("onnx_asr", types.ModuleType("onnx_asr"))
sys.modules.setdefault("onnxruntime", ort_stub)

import numpy as np  # noqa: E402

from parakeet_service import routes  # noqa: E402
from parakeet_service.config import TARGET_SR  # noqa: E402

SR = TARGET_SR
FRAME = 0.08


def grid(t, ws, jitter=0):
    """Token time on the 80 ms grid of a decode starting at ws (seconds from ws)."""
    return max(0.0, (math.floor((t - ws) / FRAME + 1e-9) + jitter) * FRAME)


def decode(truth, ws, we, skip=(), jit=None, made_up_end=False):
    """A result for a decode of [ws, we): truth words starting in it, minus
    those in `skip` (absolute intervals), on the decode's grid."""
    tokens, stamps = [], []
    for word, t in truth:
        if not (ws <= t < we):
            continue
        if any(a <= t < b for a, b in skip):
            continue
        j = jit(word, ws) if jit else 0
        tokens.append(" " + word)
        stamps.append(grid(t, ws, j))
    if made_up_end:
        tokens.append(" uh")
        stamps.append(max(0.0, (we - ws) - 0.08))
    return SimpleNamespace(text="".join(tokens).strip(), tokens=tokens, timestamps=stamps)


class Worker:
    def __init__(self, truth, **kw):
        self.truth, self.kw, self.calls = truth, kw, []

    async def submit_many(self, pieces, _key):
        out = []
        for piece in pieces:
            ws = float(piece[0]) / SR
            we = (float(piece[-1]) + 1) / SR
            self.calls.append((round(ws, 3), round(we, 3)))
            out.append(decode(self.truth, ws, we, **self.kw))
        return out


def request(worker):
    state = SimpleNamespace(worker=worker, ready=True, audio_pool=None)
    return SimpleNamespace(app=SimpleNamespace(state=state))


def samples(pairs):
    return [(int(round(a * SR)), int(round(b * SR))) for a, b in pairs]


def prepared(ranges, windows, speech, total):
    wav = np.arange(int(round(total * SR)), dtype=np.float32)
    r, w = samples(ranges), samples(windows)
    return routes._PreparedAudio(
        waveform=wav,
        ranges=r,
        windows=w,
        speech=None if speech is None else samples(speech),
        pieces=[wav[a:b] for a, b in w],
        duration=total,
    )


def run(truth, ranges, windows, speech, total, skips, redo_kw=None, first_kw=None, model="parakeet-v3:fp32"):
    """skips: per piece, absolute skip intervals for the first decode."""
    prep = prepared(ranges, windows, speech, total)
    first = [
        decode(truth, a / SR, b / SR, skip=s, **(first_kw or {})) for (a, b), s in zip(prep.windows, skips)
    ]
    worker = Worker(truth, **(redo_kw or {}))
    results = asyncio.run(routes._redo_stalled(request(worker), [prep], first, model))
    text, segments, words = routes._stitch(prep, results)
    return text, segments, words, worker.calls


def score(truth, words):
    names = [w["word"] for w in words]
    want = [w for w, _t in truth]
    missing = [w for w in want if w not in names]
    dup = sorted({w for w in names if names.count(w) > 1})
    extra = [w for w in names if w not in want]
    order = [w for w in names if w in want]
    inversions = sum(1 for a, b in zip(order, order[1:]) if want.index(a) > want.index(b))
    starts = [w["start"] for w in words]
    unsorted = sum(1 for a, b in zip(starts, starts[1:]) if b < a - 1e-9)
    return dict(missing=missing, dup=dup, extra=extra, inversions=inversions, unsorted=unsorted)
