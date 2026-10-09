import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/m75b-scripts")
from lib import *
ch_h, _ = MODS["head"]; ch_p, _ = MODS["prev"]
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
worst = []
for it in range(N):
    P = float(rng.uniform(5, 40))
    k = int(rng.integers(0, 15))
    sounds = []
    for _ in range(k):
        t = 40 + float(rng.uniform(0, P))
        d = float(rng.choice([0.03, 0.1, 0.2, 0.3, 0.45, 0.55, 0.8, 1.5, 3.0]))
        lvl = float(rng.uniform(-50, -30))
        sounds.append((t, d, lvl))
    # sometimes a quiet speaker turn
    if rng.random() < 0.3:
        t = 40 + float(rng.uniform(0, P)); d = float(rng.uniform(1, 8)); lvl = float(rng.uniform(-45, -30))
        sounds.append((t, d, lvl))
    wav = build(P, sounds, seed=it)
    rms = ch_h.frame_rms(wav)
    for ratio in (0.4, 0.6):
        h = ch_h.loud_frames(rms, ratio, 150); p = ch_p.loud_frames(rms, ratio, 150)
        extra = int((h & ~p).sum())
        if extra:
            worst.append((extra, it, ratio, P, sounds))
worst.sort(key=lambda x: -x[0])
print("cases with head-loud-not-prev frames:", len(worst))
for w in worst[:8]:
    print(w[0], "frames; it", w[1], "ratio", w[2], "P", round(w[3], 2))
    print("   ", [(round(t, 2), d, round(l, 1)) for t, d, l in sorted(w[4])])
