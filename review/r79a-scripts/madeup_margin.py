"""#68 at a stretch redo's input end: the redo makes up "uh" in its last 0.3 s,
inside the margin, where the piece heard every word (a word every 0.5 s).

python madeup_margin.py <worktree> [one|two] [offset-from-end]"""
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
off = float(sys.argv[3]) if len(sys.argv) > 3 else 0.08
phase = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
step = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
words = [(f" w{i}", round(step * i + phase, 3)) for i in range(1, int(59 / step))]


def heard(ws, we, skip=None, madeup=False):
    toks, ts = [], []
    for w, t in words:
        if ws <= t < we and not (skip and skip[0] <= t < skip[1]):
            toks.append(w)
            ts.append(round(int((t - ws) / 0.08) * 0.08, 4))
    if madeup:
        toks.append(" uh")
        ts.append(round(int((we - ws - off) / 0.08) * 0.08, 4))
    return SimpleNamespace(text="".join(toks).strip(), tokens=toks, timestamps=ts)


class Worker:
    def __init__(self):
        self.calls = []

    async def submit_many(self, pieces, _key):
        out = []
        for piece in pieces:
            ws, we = float(piece[0]) / SR, (float(piece[0]) + piece.size) / SR
            self.calls.append((round(ws, 2), round(we, 2)))
            long = we - ws >= (40 if shape == "one" else 25)
            out.append(heard(ws, we, skip=(10.0, 20.0)) if long else heard(ws, we, madeup=True))
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
print("uh in words:", [(w["word"], round(w["start"], 2)) for w in ws_ if w["word"] == "uh"])
i = text.index("w28")
print("text:", text[i:i + 45])
