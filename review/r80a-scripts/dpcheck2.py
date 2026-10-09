"""head's _quiet_cuts DP vs brute force over the same candidates, cost reimplemented independently from the stated rule."""
import os, sys, itertools, functools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import HEAD as H, SR

H.QUIET_CUT_SEARCH_SEC = 0.06
rng = np.random.default_rng(7)
bad = mism = trials = moved = 0
for trial in range(400):
    M = int(rng.uniform(1.0, 2.0) * SR)
    count = int(rng.integers(2, 4))
    L = int(rng.uniform((count - 1) * M + 1, count * M))
    start = int(rng.uniform(0, 0.5) * SR)
    end = start + L
    n = end + SR
    # blocks of random level, some deep dips
    lv = rng.uniform(0.02, 1, n // 800 + 1)
    lv[rng.random(lv.size) < 0.15] *= 0.05
    wav = (rng.standard_normal(n) * np.repeat(lv, 800)[:n]).astype(np.float32)
    even = [start + L * i // count for i in range(count + 1)]
    ls = start + int(rng.uniform(0, 0.2) * SR) if rng.random() < 0.5 else start
    ee = end - int(rng.uniform(0, 0.2) * SR) if rng.random() < 0.5 else end
    got = H._quiet_cuts(wav, even, M, ls, ee)
    reach = min(int(H.QUIET_CUT_SEARCH_SEC * SR), M // 8)
    step = H._QUIET_STEP
    def cands(p, lo, hi):
        return list(range(p - (p - lo) // step * step, p, step)) + list(range(p, hi + 1, step))
    stages = [cands(start, start, min(ls, start + reach))] + [cands(p, p - reach, p + reach) for p in even[1:-1]] + [cands(end, max(ee, end - reach), end)]
    fr = H.frame_rms(wav[start:end]).astype(np.float64) ** 2
    around = int(1.5 * SR) // H.FRAME
    def local(p):
        f = (p - start) // H.FRAME
        return float(np.median(fr[max(0, f - around): f + around + 1])) + 1e-12
    def pw(c, span, ref):
        lo, hi = max(0, c - span // 2), min(wav.size, c + span // 2)
        lo2 = max(lo, 0)
        return float(np.mean(wav[lo2:hi].astype(np.float64) ** 2)) / local(ref)
    def stage_cost(i, a):
        p = even[i]
        c = 0.01 * abs(a - p) / SR
        if i == 0 or i == len(even) - 1:
            return c
        here = pw(p, int(0.2 * SR), p); dip = pw(p, int(0.08 * SR), p); lvl = pw(a, int(0.2 * SR), p)
        if a == p:
            return c + lvl
        if min(here, dip) >= 0.5 and lvl <= here * 0.5:
            return c + lvl
        return np.inf
    best, bc = None, np.inf
    sc = [[stage_cost(i, a) for a in st] for i, st in enumerate(stages)]
    for idx in itertools.product(*[range(len(s)) for s in stages]):
        chain = [stages[i][j] for i, j in enumerate(idx)]
        if all(0 < b - a <= M for a, b in zip(chain, chain[1:])):
            c = sum(sc[i][j] for i, j in enumerate(idx))
            if c < bc - 1e-12:
                best, bc = chain, c
    gi = [stages[i].index(x) if x in stages[i] else None for i, x in enumerate(got)]
    gc = sum(sc[i][j] for i, j in enumerate(gi)) if None not in gi else np.inf
    ok = all(0 < b - a <= M for a, b in zip(got, got[1:])) and min(b - a for a, b in zip(got, got[1:])) >= M // 4
    moved += got != even
    trials += 1
    if not ok:
        bad += 1
    if best is not None and abs(gc - bc) > 1e-6 * max(1, abs(bc)):
        mism += 1
        if mism < 4:
            print("MISMATCH", trial, gc, bc, got, best)
print(f"{trials} trials ({moved} moved): bound failures {bad}, cost mismatches vs brute force {mism}")
