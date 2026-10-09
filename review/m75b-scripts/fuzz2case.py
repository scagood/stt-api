import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/m75b-scripts")
from lib import *
ch_h, _ = MODS["head"]; ch_p, _ = MODS["prev"]
seed0, target, ratio, rel = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(seed0)
for it in range(target + 1):
    n = int(rng.integers(500, 6000))
    fdb = rng.uniform(-70, -50); floor = 10 ** (fdb / 20)
    rms = floor * rng.uniform(0.85, 1.15, n)
    rms[: n // 5] = 0.1
    snd = []
    for _ in range(int(rng.integers(1, 40))):
        a = int(rng.integers(n // 5, n)); L = int(rng.choice([2, 4, 6, 10, 15, 20, 26, 30, 40, 80, 200]))
        o = rng.uniform(5, 40); lvl = floor * 10 ** (o / 20)
        rms[a: a + L] = np.maximum(rms[a: a + L], lvl * rng.uniform(0.7, 1.3, rms[a:a+L].size))
        snd.append((a, a + L, round(o, 1)))
rms = rms.astype(np.float32)
print("n", n, "floor dB", round(fdb, 1), "loud prefix to", n // 5)
print("sounds (frame a, b, dB over floor):", sorted(snd))
h = ch_h.loud_frames(rms, ratio, rel); p = ch_p.loud_frames(rms, ratio, rel)
R = lambda m: [tuple(x) for x in ch_h.runs(m).tolist()]
print("prev loud:", R(p)); print("head loud:", R(h)); print("head&~prev:", R(h & ~p)); print("prev&~head:", R(p & ~h))
