import numpy as np
import lib
lib.ROOTS["fix"] = f"{lib.S}/k78b-fix"
CF, RF = lib.load("fix")
rng = np.random.default_rng(0)
cnt = dict(main_not_pr=0, main_not_fix=0, pr_not_fix=0)
for it in range(20000):
    n = int(rng.integers(1, 1500)); kind = rng.integers(0, 4)
    if kind == 0:
        rms = np.abs(rng.standard_normal(n)) * 10 ** (rng.uniform(-80, -10) / 20)
    elif kind == 1:
        rms = np.full(n, 10 ** (rng.uniform(-80, -10) / 20))
    elif kind == 2:
        rms = 10 ** (rng.uniform(-70, -20, n) / 20)
    else:
        rms = np.full(n, 10 ** (-60 / 20)) * (1 + 0.1 * rng.random(n))
        for _ in range(rng.integers(0, 30)):
            a = int(rng.integers(0, n)); L = int(rng.integers(1, 40))
            rms[a:a + L] *= 10 ** (rng.uniform(5, 40) / 20)
    rms = rms.astype(np.float32)
    relisten = int(rng.integers(1, 200)); ratio = float(rng.choice([0.4, 0.6]))
    m = lib.CM.loud_frames(rms, ratio, relisten); p = lib.CP.loud_frames(rms, ratio, relisten); f = CF.loud_frames(rms, ratio, relisten)
    cnt["main_not_pr"] += bool((m & ~p).any()); cnt["main_not_fix"] += bool((m & ~f).any()); cnt["pr_not_fix"] += bool((p & ~f).any())
print(cnt)
