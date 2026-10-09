import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/k78a-scripts")
from lib import *
gmax = float(sys.argv[1]); seed = int(sys.argv[2]); N = int(sys.argv[3])
rng = np.random.default_rng(seed)
res = {n: [0, 0] for n in ("main", "head")}; ext = {n: 0.0 for n in ("main", "head")}
for i in range(N):
    P = float(rng.uniform(8, 40))
    ph0 = 40.5; ph1 = ph0 + float(rng.uniform(0.6, 1.2))
    sounds = [(ph0, ph1 - ph0, float(rng.uniform(-42, -36)))]; t = ph1
    while True:
        t += float(rng.uniform(0.45, gmax)); d = float(rng.uniform(0.08, 0.45))
        if t + d > 40 + P - 0.3: break
        sounds.append((t, d, float(rng.uniform(-48, -38)))); t += d
    wav = build(P, sounds, seed=i + 31 * seed)
    for n in ("main", "head"):
        r = [(a / SR, b / SR) for a, b in plan(n, wav, 0.0).ranges]
        far = [(a, b) for a, b in r if b > ph1 + 3.6 and a < 40 + P - 0.5]
        if far: res[n][0] += 1
        ps = pauses(n, wav)
        # retime: is there a non-pause past phrase end + 3.6 before the pause end?
        if not any(a <= ph1 + 3.6 and b >= 40 + P - 0.2 for a, b in ps): res[n][1] += 1
print(f"anchored gaps 0.45-{gmax}: {N} pauses; decoded >3.6 s past the phrase / retime pause broken >3.6 s past: main {res['main']} head {res['head']}")
