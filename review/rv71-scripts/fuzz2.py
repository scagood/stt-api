import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, ".")
import random, bisect
from lib import *
seed = int(sys.argv[2]); N = int(sys.argv[3])
rng = random.Random(seed)
def gen():
    seconds = rng.uniform(60, 600)
    at = rng.uniform(0, 5); sp = []
    while True:
        r = rng.random()
        L = rng.uniform(2,6) if r < .8 else rng.uniform(6,22) if r < .95 else rng.uniform(22,70)
        if at + L > seconds - 1: break
        sp.append((int(at*SR), int((at+L)*SR)))
        at += L + rng.uniform(0.25, 12)
    return sp, int(seconds*SR)
files = [gen() for _ in range(N)]
files = [f for f in files if f[0]]
own = 20*SR
def infit(sp, r):
    starts = [s for s, e in sp]; out = []
    for (a, b), (a2, b2) in zip(r, r[1:]):
        if b != a2: continue
        i = bisect.bisect_right(starts, b) - 1
        if i >= 0 and sp[i][0] < b < sp[i][1] and sp[i][1]-sp[i][0] <= own: out.append((a, b, sp[i]))
    return out
imp = reg = 0; lost = 0
for k, (sp, total) in enumerate(files):
    rn = plan(new, "v2", speech=sp, total=total).ranges
    ro = plan(old, "v2", speech=sp, total=total).ranges
    a, b = len(infit(sp, rn)), len(infit(sp, ro))
    if a < b: imp += 1
    if a > b:
        reg += 1; print("REGRESS file", k, fmt(sp)[:0])
        for x in infit(sp, rn): print("  new infit cut", x[1]/SR, "range", (x[0]/SR, x[1]/SR), "phrase", (x[2][0]/SR, x[2][1]/SR))
    for x in infit(sp, rn):
        print("residual: range before", round((x[1]-x[0])/SR,3), "cut", x[1]/SR, "phrase", (x[2][0]/SR, x[2][1]/SR))
    # coverage
    for s, e in sp:
        cov = sum(max(0, min(e, rb) - max(s, ra)) for ra, rb in rn)
        if cov < e - s: lost += 1
print("improve", imp, "regress", reg, "lost-speech segs", lost)
