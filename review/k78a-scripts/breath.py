import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/k78a-scripts")
from lib import *
gmax = float(sys.argv[1]); seed = int(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
ms = int(sys.argv[4]) if len(sys.argv) > 4 else 400
for n in MODS: MODS[n][0].VAD_MIN_SILENCE_MS = ms
rng = np.random.default_rng(seed)
res = {n: [0, 0, 0] for n in ("main", "head")}
nb = []
for i in range(N):
    P = float(rng.uniform(5, 40)); sounds = []; t = 40.3
    while True:
        t += float(rng.uniform(0.45, gmax))
        d = float(rng.uniform(0.08, 0.45))
        if t + d > 40 + P - 0.3: break
        sounds.append((t, d, float(rng.uniform(-48, -38)))); t += d
    nb.append(len(sounds))
    wav = build(P, sounds, seed=i + 7919 * seed)
    for n in ("main", "head"):
        for j, ctx in enumerate((0.0, 5.0)):
            r = plan(n, wav, ctx).ranges
            if any(b / SR > 40.5 and a / SR < 40 + P - 0.5 for a, b in r): res[n][j] += 1
        if not any(a <= 40.2 and b >= 40 + P - 0.2 for a, b in pauses(n, wav)): res[n][2] += 1
print(f"breath gaps 0.45-{gmax} s, seed {seed}, ms {ms}: {N} pauses ({sum(nb)} breaths, max {max(nb)} in one); decoded inside pause ctx0/ctx5, retime pause split: main {res['main']} head {res['head']}")
