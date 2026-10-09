import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/m75b-scripts")
from lib import *
ch_h, _ = MODS["head"]; ch_p, _ = MODS["prev"]
seed0 = int(sys.argv[1]); N = int(sys.argv[2])
rng = np.random.default_rng(seed0)
found = []
for it in range(N):
    n = int(rng.integers(500, 6000))
    floor = 10 ** (rng.uniform(-70, -50) / 20)
    rms = floor * rng.uniform(0.85, 1.15, n)
    k = 0
    # first 20% loud so file gate is high
    rms[: n // 5] = 0.1
    for _ in range(int(rng.integers(1, 40))):
        a = int(rng.integers(n // 5, n)); L = int(rng.choice([2, 4, 6, 10, 15, 20, 26, 30, 40, 80, 200]))
        lvl = floor * 10 ** (rng.uniform(5, 40) / 20)
        rms[a: a + L] = np.maximum(rms[a: a + L], lvl * rng.uniform(0.7, 1.3, rms[a:a+L].size))
    rms = rms.astype(np.float32)
    for ratio in (0.4, 0.6):
        for rel in (150, 25, 60):
            h = ch_h.loud_frames(rms, ratio, rel); p = ch_p.loud_frames(rms, ratio, rel)
            extra = int((h & ~p).sum())
            if extra:
                found.append((extra, it, ratio, rel))
found.sort(reverse=True)
print(len(found), found[:10])
