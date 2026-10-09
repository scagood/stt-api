import sys, time
sys.path.insert(0, ".")
import numpy as np
from parakeet_service import chunker, retime
chunker.VAD = "volume"; chunker.VAD_GATE_DB = None
SR = 16000; FR = 320
rng = np.random.default_rng(0)
# 3 h of frame levels directly (rms domain): loud turns, quiet turns, pauses with breaths and words
lv = []
while len(lv) < 3 * 3600 * 50:
    kind = rng.integers(0, 3)
    n = int(rng.uniform(2, 60) * 50)
    if kind == 0: seg = 10 ** (rng.uniform(-24, -18, n) / 20)
    elif kind == 1:
        seg = 10 ** (-55 / 20) * np.ones(n)
        for _ in range(int(n / 50 / 3)):
            a = rng.integers(0, n); seg[a:a + rng.integers(3, 30)] = 10 ** (-42 / 20)
    else:
        seg = 10 ** (-55 / 20) * np.ones(n)
        k = 0
        while k < n:
            d = rng.integers(5, 40); seg[k:k + d] = 10 ** (rng.uniform(-46, -40) / 20); k += d + rng.integers(10, 60)
    lv.extend(seg)
rms = np.asarray(lv, np.float32) * (1 + 0.1 * rng.standard_normal(len(lv))).astype(np.float32)
t = time.perf_counter(); chunker.loud_frames(rms, 0.4, 150); t1 = time.perf_counter() - t
t = time.perf_counter(); chunker.loud_frames(rms, 0.6, 150); t2 = time.perf_counter() - t
print(f"3 h of frames: loud_frames 0.4 {t1:.2f}s, 0.6 {t2:.2f}s")
# pathological: one loud blip then 1 h of fragmented quiet speech
q = np.full(3600 * 50, 10 ** (-55 / 20), np.float32)
k = 0
while k < q.size:
    d = rng.integers(5, 30); q[k:k + d] = 10 ** (-40 / 20); k += d + rng.integers(20, 200)
q[:10] = 0.5
t = time.perf_counter(); chunker.loud_frames(q, 0.4, 150); print(f"1 h fragmented quiet: {time.perf_counter() - t:.2f}s")
