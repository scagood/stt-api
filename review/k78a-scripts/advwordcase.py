import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/k78a-scripts")
from lib import *
import diff as D
seed = int(sys.argv[1]); target = int(sys.argv[2])
rng = np.random.default_rng(seed)
for it in range(target + 1):
    floor = float(rng.uniform(-62, -50))
    P = float(rng.uniform(6, 30))
    sounds = []
    t = 40 + float(rng.uniform(0.3, 3))
    ph_end = None
    for _ in range(int(rng.integers(1, 4))):
        d = float(rng.uniform(0.5, 3.0)); lvl = float(rng.uniform(-36, -31))
        if t + d > 40 + P - 0.5: break
        sounds.append((t, d, lvl)); ph_end = t + d
        t += d + float(rng.uniform(0.1, 1.0))
    if ph_end is None: continue
    w0 = ph_end + 3.0 - float(rng.uniform(0.02, 0.4)); wd = float(rng.uniform(0.1, 0.48))
    sounds.append((w0, wd, float(rng.uniform(-40, -31))))
    t = w0 + wd + float(rng.uniform(0.05, 1.0))
    while t < 40 + P - 0.2:
        d = float(rng.uniform(0.1, 1.2)); lvl = floor + float(rng.uniform(6, 16))
        d = min(d, 40 + P - t)
        sounds.append((t, d, lvl))
        t += d + float(rng.uniform(0.05, 2.5))
wav = build(P, sounds, seed=target + 100000 * seed, floor_db=floor)
words = [(s, s + d) for s, d, l in sounds if d >= 0.12]
print("floor", round(floor, 1), "pause 40 ..", round(40 + P, 2))
for s in sounds: print("  sound %.2f-%.2f %.1f dBFS (+%.1f)" % (s[0], s[0] + s[1], s[2], s[2] - floor))
for n in ("main", "head"):
    p = plan(n, wav, 0.0)
    print(n, "speech", [x for x in secs(p.speech) if x[1] > 39 and x[0] < 41 + P])
    print(n, "ranges", secs(p.ranges), "score", D.score_plan(p.ranges, words))
