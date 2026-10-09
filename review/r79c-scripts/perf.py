"""Worst-case splice cost: a piece hearing short bursts between 3.3 s gaps, so
stretch windows merge into one; redo hears every word. WT=<wt> python perf.py [rate] [secs]"""
import asyncio, logging, sys, time
from types import SimpleNamespace
import lib
from lib import routes, S
logging.disable(logging.CRITICAL)
rate = float(sys.argv[1]) if len(sys.argv) > 1 else 4.0
total = float(sys.argv[2]) if len(sys.argv) > 2 else 75.0
words = [(f" w{i}", round(0.1 + i / rate, 3)) for i in range(int((total - 0.5) * rate))]
def heard(origin, ws):
    return SimpleNamespace(text="", tokens=[t for t, _ in ws], timestamps=[round(a - origin, 4) for _, a in ws])
first = [(t, a) for t, a in words if (a % 4.8) < 1.5]
class W:
    def __init__(self): self.asked = []
    async def submit_many(self, pieces, key):
        out = []
        for p in pieces:
            a = float(p[0]) / 16000; b = a + p.size / 16000
            self.asked.append((round(a, 2), round(b, 2)))
            out.append(heard(a, [(t, x) for t, x in words if a <= x < b]))
        return out
routes.speech_segments = lambda wav: [(0, wav.size)]
prep = lib.prepared(total, [(0, total)], [(0, total)], None)
w = W()
t0 = time.perf_counter()
res = asyncio.run(routes._redo_stalled(lib.request(w), [prep], [heard(0.0, first)], "parakeet-v3:fp32"))
dt = time.perf_counter() - t0
text, segs, ws = routes._stitch(prep, res)
print(f"rate={rate}/s total={total}s first={len(first)} words, redo={w.asked}, out={len(ws)}/{len(words)} words, _redo_stalled {dt*1000:.0f} ms")
