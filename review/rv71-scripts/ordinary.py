import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, ".")
import numpy as np
from lib import *
def ordinary(seed, seconds=600):
    rng = np.random.default_rng(seed); sp = []; a = 1.0
    while True:
        L = rng.uniform(2, 6)
        if a + L > seconds - 1: return sp, int(seconds*SR)
        sp.append((int(a*SR), int((a+L)*SR))); a += L + rng.uniform(0.5, 1.5)
same_old = {m: 0 for m in BOUNDS}; same_main = {m: 0 for m in BOUNDS}; shrink = 0; small = 0; m0 = 0
fix1same = 0
for seed in range(300):
    sp, total = ordinary(seed)
    for m in BOUNDS:
        rn = plan(new, m, speech=sp, total=total).ranges
        same_old[m] += rn == plan(old, m, speech=sp, total=total).ranges
        same_main[m] += rn == plan(main, m, speech=sp, total=total).ranges
        if m == "v2":
            mg = (rn[-1][1] - sp[-1][1]) / SR
            if mg < 3 - 1e-6 and rn[-1][1] < total: shrink += 1
            if mg < 0.5: small += 1
            if mg == 0: m0 += 1
            fix1same += rn == plan(fix1, m, speech=sp, total=total).ranges
print("same as old", same_old, "same as main", same_main)
print("v2 margin shrunk", shrink, "<0.5", small, "==0", m0, "fix1 same", fix1same)
