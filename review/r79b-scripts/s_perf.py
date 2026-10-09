import time, logging, tracemalloc
import lib
from lib import w, run, routes
logging.disable(logging.CRITICAL)
# One 75 s piece; the first decode hears 2.5 s bursts with 3.4 s gaps (stretches margins overlap),
# the redo hears every word (4 words/s, short repeated function words).
total = 75.0
truth = []
t, i = 0.1, 0
while t < total - 0.2:
    truth.append(w(["the", "a", "and", "I", "no"][i % 5] if i % 3 else f"w{i}", round(t, 3)))
    t += 0.25; i += 1
gaps = []
g = 2.5
while g < total:
    gaps.append((g, g + 3.4)); g += 5.9
def beh(a, b, phase, k):
    return dict(skip=gaps) if phase == "first" else {}
tracemalloc.start()
t0 = time.time()
r = run(total, [(0, total)], [(0, total)], None, lib.Model(truth, beh), vad=[(0, total)])
print("one 75 s piece:", round(time.time() - t0, 2), "s; peak MB", round(tracemalloc.get_traced_memory()[1] / 1e6, 1), "calls", [c for c in r.calls if c[0] != "first"], "words", len(r.words), "of", len(truth))
