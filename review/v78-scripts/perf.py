import time, numpy as np
import lib_b as lib
rng = np.random.default_rng(0)
# 2 h of rms: speech turns, long quiet stretches with sparse sounds
n = 2 * 3600 * 50
lv = np.full(n, -55.0) + rng.normal(0, 1, n)
i = 0
while i < n:
    L = int(rng.integers(250, 2000)); lv[i:i + L] = -20 + rng.normal(0, 3, min(L, n - i)); i += L
    Q = int(rng.integers(100, 1500)); j = i
    while j < i + Q and j < n:
        k = int(rng.integers(2, 40)); lv[j:j + k] = -55 + rng.uniform(5, 25); j += k + int(rng.integers(5, 80))
    i += Q
rms = (10 ** (lv / 20)).astype(np.float32)
for tag in ("main", "old", "head"):
    c = lib.MODS[tag][0]
    best = 1e9
    for _ in range(3):
        t = time.perf_counter(); out = c.loud_frames(rms, 0.4, 150); best = min(best, time.perf_counter() - t)
    print(tag, f"{best:.3f}s", int(out.sum()))
