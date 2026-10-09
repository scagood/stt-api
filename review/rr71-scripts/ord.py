import collections
from h import *
import fuzz
# 1) silent-range classification on new (seed 1 layouts)
fuzz.rng = np.random.default_rng(1)
C = collections.Counter()
for b in ("v2", "v3", "wh"):
    for _ in range(3000):
        sp, seconds = fuzz.layout()
        if not sp: continue
        spS = [tuple(int(x * SR) for x in s) for s in sp]; total = int(seconds * SR)
        p = plan(new, spS, total, BOUNDS[b])
        if not p.speech: continue
        for i, (s, e) in enumerate(p.ranges):
            if not any(a < e and s < z for a, z in p.speech):
                last = i == len(p.ranges) - 1 and s >= p.speech[-1][1]
                C[(b, "trailing-margin" if last else "other")] += 1
print("silent on new:", dict(C))
# 2) ordinary speech (2-6 s phrases, 0.5-1.5 s pauses): new vs main identical?
diff = collections.Counter()
for seed in range(300):
    r = np.random.default_rng(seed); speech, at = [], 1.0
    while True:
        L = r.uniform(2, 6)
        if at + L > 599: break
        speech.append((int(at*SR), int((at+L)*SR))); at += L + r.uniform(0.5, 1.5)
    for b in ("v2", "v3", "wh"):
        rs = {v: plan(VERS[v], speech, 600*SR, BOUNDS[b]).ranges for v in VERS}
        diff[(b, "new!=main")] += rs["new"] != rs["main"]; diff[(b, "new!=old")] += rs["new"] != rs["old"]
print("ordinary speech files differing (of 300):", dict(diff))
