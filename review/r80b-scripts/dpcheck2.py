"""Head's _quiet_cuts DP vs brute force over the same candidates and its own rule."""
import os, sys, itertools, functools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from synth import PR, SR

PR.QUIET_CUT_SEARCH_SEC = float(os.environ.get("REACH", "0.05"))
rng = np.random.default_rng(int(os.environ.get("SEED", "0")))
bad = moved_tot = 0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
for trial in range(N):
    M = int(rng.uniform(1.0, 2.0) * SR)
    count = int(rng.integers(2, 4))
    L = int(rng.uniform((count - 1) * M + 1, count * M))
    start = int(rng.uniform(0, 0.5) * SR)
    end = start + L
    n = end + SR
    wav = (rng.standard_normal(n) * np.repeat(10 ** (rng.uniform(-40, 0, n // 800 + 1) / 20), 800)[:n]).astype(np.float32)
    even = [start + L * i // count for i in range(count + 1)]
    ls = start + int(rng.uniform(0, 0.2) * SR) if rng.random() < 0.5 else start
    ee = end - int(rng.uniform(0, 0.2) * SR) if rng.random() < 0.5 else end
    got = PR._quiet_cuts(wav, even, M, ls, ee)
    reach = min(int(PR.QUIET_CUT_SEARCH_SEC * SR), M // 8)
    step = PR._QUIET_STEP

    def cands(p, lo, hi):
        return list(range(p - (p - lo) // step * step, p, step)) + list(range(p, hi + 1, step))

    stages = [cands(start, start, min(ls, start + reach))] + [cands(p, p - reach, p + reach) for p in even[1:-1]] + [cands(end, max(ee, end - reach), end)]
    frames = PR.frame_rms(wav[start:end]).astype(np.float64) ** 2
    around = int(PR._LOCAL_SEC * SR) // PR.FRAME

    def local(p):
        f = (p - start) // PR.FRAME
        return float(np.median(frames[max(0, f - around): f + around + 1])) + 1e-12

    def lvl(c, p, span):
        lo, hi = max(0, c - span // 2), min(wav.size, c + span // 2)
        return float(np.sum(wav[lo:hi].astype(np.float64) ** 2)) / max(hi - lo, 1) / local(p)

    stage_cost = []
    for k, (p, st) in enumerate(zip(even, stages)):
        if k == 0 or k == len(even) - 1:
            stage_cost.append({c: 0.01 * abs(c - p) / SR for c in st})
            continue
        here, dip = lvl(p, p, PR._QUIET_SPAN), lvl(p, p, PR._DIP_SPAN)
        d = {}
        for c in st:
            l = lvl(c, p, PR._QUIET_SPAN)
            if c == p or (min(here, dip) >= PR._QUIET and l <= here * PR._QUIETER):
                d[c] = 0.01 * abs(c - p) / SR + l
        stage_cost.append(d)

    def cost(chain):
        tot = 0.0
        for d, c in zip(stage_cost, chain):
            if c not in d:
                return np.inf
            tot += d[c]
        return tot

    best, bc = None, np.inf
    for chain in itertools.product(*[sorted(d) for d in stage_cost]):
        if all(0 < b - a <= M for a, b in zip(chain, chain[1:])):
            c = cost(chain)
            if c < bc - 1e-12:
                best, bc = chain, c
    gc = cost(got)
    ok = all(0 < b - a <= M for a, b in zip(got, got[1:])) and min(b - a for a, b in zip(got, got[1:])) >= M // 4
    ok = ok and abs(gc - bc) <= 1e-9 * max(1, bc)
    moved_tot += got != even
    if not ok:
        bad += 1
        if bad < 4:
            print("MISMATCH", trial, count, gc, bc, got, best)
print(f"{N - bad}/{N} trials: DP chain feasible, parts in [max/4, max], cost == brute force (chains moved: {moved_tot})")
