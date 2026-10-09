"""A word the piece heard as one token ("today" at 9.0 s), which the stretch
redo hears as two words ("to" 9.0, "day" 9.24), in the redo's margin.

python split_dup.py <worktree> [one|two]"""
import asyncio
import logging
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

logging.basicConfig(level=logging.WARNING, format="   log: %(message)s")
shape = sys.argv[2] if len(sys.argv) > 2 else "one"
gap = float(sys.argv[3]) if len(sys.argv) > 3 else 0.24

# truth: a word every 0.5 s; "today" at 9.0 s; the piece skips 10-20 s
words = [(f" w{i}", 0.5 * i) for i in range(1, 18)] + [(" today", 9.0)] + [(f" v{i}", 9.5 + 0.5 * i) for i in range(1, 100)]
words = [(w, t) for w, t in words if t < 59.5]


def heard(ws, we, skip=None, split=False):
    toks, ts = [], []
    for w, t in words:
        if not (ws <= t < we) or (skip and skip[0] <= t < skip[1]):
            continue
        if split and w == " today":
            toks += [" to", " day"]
            ts += [t - ws, t - ws + gap]
        else:
            toks.append(w)
            ts.append(round(t - ws, 4))
    return SimpleNamespace(text="".join(toks).strip(), tokens=toks, timestamps=ts)


class Worker:
    def __init__(self):
        self.calls = []

    async def submit_many(self, pieces, _key):
        out = []
        for piece in pieces:
            ws, we = float(piece[0]) / SR, (float(piece[0]) + piece.size) / SR
            self.calls.append((round(ws, 2), round(we, 2)))
            # long decodes skip 10-20 s again; short ones hear it, and hear "today" as "to day"
            out.append(heard(ws, we, skip=(10.0, 20.0)) if we - ws >= (40 if shape == "one" else 25) else heard(ws, we, split=True))
        return out


wav = np.arange(60 * SR, dtype=np.float64)
if shape == "one":
    ranges = windows = [(0, 60 * SR)]
    speech = None
else:
    ranges, windows = [(0, 30 * SR), (30 * SR, 60 * SR)], [(0, 35 * SR), (25 * SR, 60 * SR)]
    speech = [(0, 60 * SR)]
routes.speech_segments = lambda w: [(0, w.size)]
prep = routes._PreparedAudio(waveform=wav, ranges=ranges, windows=windows, speech=speech,
                             pieces=[wav[a:b] for a, b in windows], duration=60.0)
first = [heard(a / SR, b / SR, skip=(10.0, 20.0)) for a, b in windows]
worker = Worker()
req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(worker=worker, ready=True, audio_pool=None)))
results = asyncio.run(routes._redo_stalled(req, [prep], first, "parakeet-v3:fp32"))
text, segs, ws_ = routes._stitch(prep, results)
print("redo calls:", worker.calls)
i = text.index("w16")
print("text:", text[i:i + 60])
print("words near 9 s:", [(w["word"], round(w["start"], 2)) for w in ws_ if 8.4 < w["start"] < 10.6])
