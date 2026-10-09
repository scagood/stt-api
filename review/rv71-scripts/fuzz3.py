import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, ".")
import random, statistics
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
silent = []; kinds = {}
for sp, total in files:
    p = plan(new, "v2", speech=sp, total=total); r = p.ranges
    for i, (a, b) in enumerate(r):
        if not any(s < b and e > a for s, e in sp):
            trailing = i == len(r) - 1 or r[i+1][0] != b and not any(s < r[i+1][1] and e > r[i+1][0] for s, e in sp[:0])
            prev_margin = a - max(e for s, e in sp if e <= b) if i and r[i-1][1] == a else None
            silent.append((i == len(r) - 1, prev_margin, (b - a) / SR, p.windows[i], (a, b)))
print("silent", len(silent), "last-range", sum(1 for x in silent if x[0]))
pm = [x[1]/SR for x in silent if x[1] is not None]
print("prev margin >=0.5:", sum(1 for m in pm if m >= 0.5), "of", len(pm), "median", statistics.median(pm) if pm else None)
print("silent len median", statistics.median(x[2] for x in silent))
print("not last:", [(fmt([x[4]]), x[1]) for x in silent if not x[0]][:5])
p = plan(new, "v2", [(3, 37.5)], 60); print(fmt(p.ranges), fmt(p.windows))
