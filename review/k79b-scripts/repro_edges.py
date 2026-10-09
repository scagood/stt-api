"""Repro: words at a skipped stretch's edges are dropped by the PR's splice.

A two-piece clip (cut at 60 s, 5 s of context, speech throughout). Piece 0's
first decode hears "... we were understanding" (to 19.84 s), skips 20.0-27.99 s,
and resumes at "Rome" (28.0 s). Every redo hears every word in its input
(a fake Parakeet: tokens on its own 80 ms grid from its input's start).

"the" (20.08 s) is 0.24 s after "ing", the piece's last token before the stretch;
"of" (27.76 s) is 0.24 s before "Rome", its first token after it.

python repro_edges.py <worktree>"""
from __future__ import annotations

import asyncio
import logging
import math
import sys
import types
from types import SimpleNamespace

sys.path.insert(0, sys.argv[1])
ort = types.ModuleType("onnxruntime")
ort.get_available_providers = lambda: ["CPUExecutionProvider"]
sys.modules.setdefault("onnx_asr", types.ModuleType("onnx_asr"))
sys.modules.setdefault("onnxruntime", ort)
import numpy as np  # noqa: E402

from parakeet_service import routes  # noqa: E402
from parakeet_service.config import TARGET_SR as SR  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(message)s")

# (token, absolute start time)
TOKENS = [(f" w{i}", 0.5 * i) for i in range(1, 39)]  # 0.5 .. 19.0
TOKENS += [(" we", 19.2), (" were", 19.36), (" under", 19.52), ("stand", 19.68), ("ing", 19.84)]
TOKENS += [(" the", 20.08), (" history", 20.4)]
TOKENS += [(f" s{i}", 21.0 + 0.5 * i) for i in range(13)]  # 21.0 .. 27.0
TOKENS += [(" part", 27.4), (" of", 27.76), (" Rome", 28.0), (".", 28.3)]
TOKENS += [(f" v{i}", 28.8 + 0.5 * i) for i in range(183)]  # 28.8 .. 119.8
SKIP = (20.0, 27.99)


def decode(ws, we, skip=None):
    toks, ts = [], []
    for tok, t in TOKENS:
        if ws <= t < we and not (skip and skip[0] <= t < skip[1]):
            toks.append(tok)
            ts.append(math.floor((t - ws) / 0.08 + 1e-6) * 0.08)
    return SimpleNamespace(text="".join(toks).strip(), tokens=toks, timestamps=ts)


class Worker:
    def __init__(self):
        self.calls = []

    async def submit_many(self, pieces, _key):
        out = []
        for piece in pieces:
            ws, we = float(piece[0]) / SR, (float(piece[-1]) + 1) / SR
            self.calls.append((round(ws, 2), round(we, 2)))
            out.append(decode(ws, we))
        return out


wav = np.arange(120 * SR, dtype=np.float32)
ranges = [(0, 60 * SR), (60 * SR, 120 * SR)]
windows = [(0, 65 * SR), (55 * SR, 120 * SR)]
prepared = routes._PreparedAudio(
    waveform=wav, ranges=ranges, windows=windows, speech=[(0, 120 * SR)],
    pieces=[wav[a:b] for a, b in windows], duration=120.0,
)
first = [decode(0, 65, SKIP), decode(55, 120)]
worker = Worker()
state = SimpleNamespace(worker=worker, ready=True, audio_pool=None)
request = SimpleNamespace(app=SimpleNamespace(state=state))
results = asyncio.run(routes._redo_stalled(request, [prepared], first, "parakeet-v3:fp32"))
text, segments, words = routes._stitch(prepared, results)
print("redo inputs (s):", worker.calls)
i = text.index("were")
print("text:", text[i - 3 : i + 160])
j = text.index("part")
print("text:", text[j - 20 : j + 20])
names = [w["word"] for w in words]
for key in ("the", "of", "Rome."):
    print(f"  {key!r}: {names.count(key)} in words, {text.split().count(key)} in text")
