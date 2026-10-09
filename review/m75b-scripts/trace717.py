import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/m75b-scripts")
from trace import *
rng = np.random.default_rng(2)
for it in range(718):
    n = int(rng.integers(500, 6000))
    fdb = rng.uniform(-70, -50); floor = 10 ** (fdb / 20)
    rms = floor * rng.uniform(0.85, 1.15, n)
    rms[: n // 5] = 0.1
    for _ in range(int(rng.integers(1, 40))):
        a = int(rng.integers(n // 5, n)); L = int(rng.choice([2, 4, 6, 10, 15, 20, 26, 30, 40, 80, 200]))
        o = rng.uniform(5, 40); lvl = floor * 10 ** (o / 20)
        rms[a: a + L] = np.maximum(rms[a: a + L], lvl * rng.uniform(0.7, 1.3, rms[a:a+L].size))
rms = rms.astype(np.float32)
for mode in ("prev", "head"):
    m = trace(rms, 0.6, 150, mode)
    assert (m == MODS[mode][0].loud_frames(rms, 0.6, 150)).all()
