"""_quiet_cuts' DP vs brute force over the same candidates (small reach), incl. movable ends."""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from synth import PR, SR

PR.QUIET_CUT_SEARCH_SEC = 0.05  # 11 candidates a stage
rng = np.random.default_rng(0)
bad = 0
for trial in range(300):
    M = int(rng.uniform(1.0, 2.0) * SR)
    count = int(rng.integers(2, 4))
    L = int(rng.uniform((count - 1) * M + 1, count * M))
    start = int(rng.uniform(0, 0.5) * SR)
    end = start + L
    wav = (rng.standard_normal(end + SR) * np.repeat(rng.uniform(0.01, 1, (end + SR) // 800 + 1), 800)[: end + SR]).astype(np.float32)
    even = [start + L * i // count for i in range(count + 1)]
    ls = start + int(rng.uniform(0, 0.2) * SR) if rng.random() < 0.5 else start
    ee = end - int(rng.uniform(0, 0.2) * SR) if rng.random() < 0.5 else end
    got = PR._quiet_cuts(wav, even, M, ls, ee)
    # brute force with the same candidates and costs
    reach = min(int(PR.QUIET_CUT_SEARCH_SEC * SR), M // 8)
    step = PR._QUIET_STEP
    def cands(p, lo, hi):
        return list(range(p - (p - lo) // step * step, p, step)) + list(range(p, hi + 1, step))
    stages = [cands(start, start, min(ls, start + reach))] + [cands(p, p - reach, p + reach) for p in even[1:-1]] + [cands(end, max(ee, end - reach), end)]
    rms = PR.frame_rms(wav[start:end]); scale = 1 / (float(np.median(rms)) ** 2 + 1e-12)
    import functools
    @functools.lru_cache(None)
    def pw(c):
        lo, hi = max(0, c - PR._QUIET_SPAN // 2), min(wav.size, c + PR._QUIET_SPAN // 2)
        lv = float(np.mean(wav[lo:hi].astype(np.float64) ** 2)) * scale
        return lv if lv < PR._QUIET else 1.0
    def cost(chain):
        c = sum(PR._QUIET_CUT_COST_PER_SEC * abs(a - p) / SR for a, p in zip(chain, even))
        return c + sum(pw(a) for a in chain[1:-1])
    best, bc = None, np.inf
    for chain in itertools.product(*stages):
        if all(0 < b - a <= M for a, b in zip(chain, chain[1:])):
            c = cost(chain)
            if c < bc - 1e-12:
                best, bc = chain, c
    gc = cost(got)
    ok = all(0 < b - a <= M for a, b in zip(got, got[1:])) and min(b - a for a, b in zip(got, got[1:])) >= M // 4
    if best is None:
        ok = ok and got == even
    elif gc > bc + 1e-9:
        ok = False
    if not ok:
        bad += 1
        if bad < 4:
            print("MISMATCH", trial, count, gc, bc, got, best)
print(f"{300 - bad}/300 trials: DP chain feasible, every part in [max/4, max], cost equal to brute force")
