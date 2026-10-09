"""Real model end to end: a 30 s one-piece clip through head's _redo_stalled
(real volume VAD, real redo decodes) and _stitch. Usage: WT=... real_e2e.py name off"""
import asyncio, os, sys, json, logging
from types import SimpleNamespace
sys.path.insert(0, os.environ["WT"])
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import realmodel as rm
from parakeet_service import routes
from parakeet_service.config import TARGET_SR as SR
logging.basicConfig(level=logging.WARNING)
name, off = sys.argv[1], float(sys.argv[2])
CLIP = 30.0
wav = rm.audio(name)[int(off * SR): int((off + CLIP) * SR)].copy()
first = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
class Worker:
    def __init__(self): self.calls = []
    async def submit_many(self, pieces, key):
        self.calls += [(round(float(np.searchsorted(np.arange(1), 0)), 2), round(p.size / SR, 2)) for p in pieces]
        return [rm.recognize(np.ascontiguousarray(p, dtype=np.float32)) for p in pieces]
prep = routes._PreparedAudio(waveform=wav, ranges=[(0, wav.size)], windows=[(0, wav.size)], speech=None, pieces=[wav], duration=CLIP)
w = Worker()
req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(worker=w, ready=True, audio_pool=None)))
async def go():
    res = [first_res] if first else await w.submit_many(prep.pieces, "k")
    main_text = routes._stitch(prep, res)[0]
    out = await routes._redo_stalled(req, [prep], res, "parakeet-v3:int8")
    return main_text, out
first_res = None
if first:
    first_res = SimpleNamespace(text=routes._decoded(first["ptoks"]), tokens=first["ptoks"], timestamps=first["pts"])
main_text, out = asyncio.run(go())
text, segs, words = routes._stitch(prep, out)
print("VAD speech:", [(round(a / SR, 2), round(b / SR, 2)) for a, b in routes.speech_segments(wav)][:12])
print("redo audio lengths:", w.calls)
print("MAIN:", main_text)
print("HEAD:", text)
names = [x["word"] for x in words]
print("text==words", text.split() == names, "monotonic", all(b["start"] >= a["start"] for a, b in zip(words, words[1:])))
print("WORDS:", [(x["word"], round(x["start"], 2)) for x in words])
print("FIRST tokens:", list(zip(first_res.tokens, [round(t, 2) for t in first_res.timestamps]))[:60])
print("OUT tokens:", list(zip(out[0].tokens, [round(t, 2) for t in out[0].timestamps]))[:80])
