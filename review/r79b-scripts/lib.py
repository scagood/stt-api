"""Fake-Parakeet harness driving the real routes._redo_stalled and _stitch.

Import after setting WT (worktree root) in the environment."""
from __future__ import annotations

import asyncio
import math
import os
import sys
import types
from types import SimpleNamespace

WT = os.environ["WT"]
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
HEAD = hasattr(routes, "_redo_stretches")
PHASE = ["first"]


def _wrap(name):
    orig = getattr(routes, name)

    async def wrapped(*a, **k):
        PHASE[0] = name
        return await orig(*a, **k)

    setattr(routes, name, wrapped)


if HEAD:
    _wrap("_redo_ranges")
    _wrap("_redo_stretches")


def S(x):
    return int(round(x * SR))


class W:
    """A truth word: tokens [(text, abs time)], text[0] starts with a space."""

    def __init__(self, toks):
        self.toks = toks

    @property
    def t(self):
        return self.toks[0][1]

    @property
    def name(self):
        return "".join(x for x, _ in self.toks).strip()


def w(name, t, ntok=1, step=0.08):
    if ntok == 1:
        return W([(" " + name, t)])
    parts = [name[: max(1, len(name) // ntok)]]
    rest = name[len(parts[0]):]
    per = max(1, -(-len(rest) // max(1, ntok - 1)))
    while rest:
        parts.append(rest[:per])
        rest = rest[per:]
    return W([((" " if i == 0 else "") + p, round(t + i * step, 4)) for i, p in enumerate(parts)])


def rel(t, a, j=0):
    return max(0.0, (math.floor((t - a) / FRAME + 1e-9) + j) * FRAME)


class Model:
    """beh(a, b, phase, idx) -> dict(skip=[(lo, hi)], jit=fn(i)->frames, alt={i: [(text, dt)]},
    extras=[(text, abs t)], frag=bool) ; idx is the call number within phase."""

    def __init__(self, truth, beh=lambda a, b, phase, idx: {}):
        self.truth, self.beh = truth, beh
        self.calls = []

    def decode(self, a, b, phase, idx):
        self.calls.append((phase, round(a, 3), round(b, 3)))
        o = self.beh(a, b, phase, idx) or {}
        skip, jit, alt, extras, frag = (o.get("skip", ()), o.get("jit"), o.get("alt", {}), o.get("extras", ()),
                                         o.get("frag", False))
        out = []  # (time, text)
        for i, word in enumerate(self.truth):
            if any(lo <= word.t < hi for lo, hi in skip):
                continue
            j = jit(i) if jit else 0
            toks = word.toks
            if i in alt:
                if not alt[i]:
                    continue
                toks = [(x, word.t + dt) for x, dt in alt[i]]
            if a <= toks[0][1] < b:
                for x, t in toks:
                    if t < b or not frag:
                        out.append((rel(t, a, j), x))
            elif frag and toks[0][1] < a and any(a <= t < b for _x, t in toks):
                first = True
                for x, t in toks:
                    if a <= t < b:
                        out.append((rel(t, a, j), (" " + x.strip()) if first else x))
                        first = False
        for x, t in extras:
            if a <= t < b:
                out.append((rel(t, a), x))
        out.sort(key=lambda p: p[0])
        tokens = [x for _t, x in out]
        times = [round(t, 4) for t, _x in out]
        return SimpleNamespace(text=routes._decoded(tokens) if hasattr(routes, "_decoded") else
                               "".join(tokens).strip(), tokens=tokens, timestamps=times)


class Worker:
    def __init__(self, model):
        self.model = model
        self.counts = {}

    async def submit_many(self, pieces, _key):
        phase = PHASE[0]
        out = []
        for piece in pieces:
            a = float(piece[0]) / SR
            b = a + piece.size / SR
            idx = self.counts.get(phase, 0)
            self.counts[phase] = idx + 1
            out.append(self.model.decode(a, b, phase, idx))
        return out


def request(worker, pool=None):
    state = SimpleNamespace(worker=worker, ready=True, audio_pool=pool, align_pool=pool)
    return SimpleNamespace(app=SimpleNamespace(state=state))


def prepared(total, ranges, windows, speech):
    wav = np.arange(S(total), dtype=np.float64)
    r = [(S(a), S(b)) for a, b in ranges]
    wi = [(S(a), S(b)) for a, b in windows]
    sp = None if speech is None else [(S(a), S(b)) for a, b in speech]
    return routes._PreparedAudio(waveform=wav, ranges=r, windows=wi, speech=sp,
                                 pieces=[wav[a:b] for a, b in wi], duration=total)


def run(total, ranges, windows, speech, model, vad=None, model_key="parakeet-v3:fp32", **stitch):
    if vad is not None and hasattr(routes, "speech_segments"):
        routes.speech_segments = lambda wav: [(S(a), S(b)) for a, b in vad]
    prep = prepared(total, ranges, windows, speech)
    worker = Worker(model)
    req = request(worker)

    async def go():
        PHASE[0] = "first"
        results = await worker.submit_many(prep.pieces, model_key)
        PHASE[0] = "_redo_ranges"
        return await routes._redo_stalled(req, [prep], results, model_key)

    results = asyncio.run(go())
    text, segments, words = routes._stitch(prep, results, **stitch)
    return SimpleNamespace(text=text, segments=segments, words=words, calls=model.calls, prep=prep, results=results)


def two(total=120.0, cut=60.0, ctx=5.0):
    return [(0.0, cut), (cut, total)], [(0.0, cut + ctx), (cut - ctx, total)]


def three(total=180.0, c1=60.0, c2=120.0, ctx=5.0):
    return [(0.0, c1), (c1, c2), (c2, total)], [(0.0, c1 + ctx), (c1 - ctx, c2 + ctx), (c2 - ctx, total)]


def filler(start, stop, step=0.4, prefix="w", ntok=1):
    out, t, i = [], start, 0
    while t < stop - 1e-9:
        out.append(w(f"{prefix}{i}", round(t, 4), ntok))
        t += step
        i += 1
    return out


def names(words):
    return [x["word"] for x in words]


def check(words, text):
    """Basic invariants: sorted, text agrees with words."""
    starts = [x["start"] for x in words]
    unsorted = sum(1 for a, b in zip(starts, starts[1:]) if b < a - 1e-9)
    bad_span = sum(1 for x in words if x["end"] < x["start"] - 1e-9)
    mismatch = text.split() != names(words)
    return dict(unsorted=unsorted, bad_span=bad_span, mismatch=mismatch)
