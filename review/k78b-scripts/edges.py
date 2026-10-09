"""Random rms arrays straight into loud_frames: crashes, dtype/shape, first-level loud kept,
and how often PR drops a frame main marks loud (and vice versa)."""
import numpy as np
import lib
rng = np.random.default_rng(0)
pr_drops = main_drops = 0
examples = []
for it in range(20000):
    n = int(rng.integers(1, 1500))
    kind = rng.integers(0, 4)
    if kind == 0:
        rms = np.abs(rng.standard_normal(n)) * 10 ** (rng.uniform(-80, -10) / 20)
    elif kind == 1:
        rms = np.full(n, 10 ** (rng.uniform(-80, -10) / 20))
    elif kind == 2:
        rms = 10 ** (rng.uniform(-70, -20, n) / 20)
    else:  # floor with bursts
        rms = np.full(n, 10 ** (-60 / 20)) * (1 + 0.1 * rng.random(n))
        for _ in range(rng.integers(0, 30)):
            a = int(rng.integers(0, n)); L = int(rng.integers(1, 40))
            rms[a:a + L] *= 10 ** (rng.uniform(5, 40) / 20)
    rms = rms.astype(np.float32)
    relisten = int(rng.integers(1, 200))
    ratio = float(rng.choice([0.4, 0.6]))
    m = lib.CM.loud_frames(rms, ratio, relisten)
    p = lib.CP.loud_frames(rms, ratio, relisten)
    assert p.dtype == bool and p.shape == rms.shape
    first = rms > lib.CP.relative_gate(rms, ratio)
    assert (p | ~first).all()
    if (m & ~p).any():
        pr_drops += 1
        if len(examples) < 5:
            examples.append((it, n, int(kind), relisten, ratio, int((m & ~p).sum()), int((p & ~m).sum())))
    if (p & ~m).any():
        main_drops += 1
print("arrays where main loud frames are quiet on PR:", pr_drops, " PR loud quiet on main:", main_drops)
print(examples)
