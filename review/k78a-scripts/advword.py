import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/k78a-scripts")
from lib import *
import diff as D
seed = int(sys.argv[1]); N = int(sys.argv[2])
rng = np.random.default_rng(seed)
worse_rt = []; worse_pl = []; better_rt = better_pl = 0; tot = {"main": [0, 0], "head": [0, 0]}; n = 0
for it in range(N):
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
    wav = build(P, sounds, seed=it + 100000 * seed, floor_db=floor)
    words = [(s, s + d) for s, d, l in sounds if d >= 0.12]
    n += 1
    r = {}
    for name in ("main", "head"):
        k, c, dr = D.score_plan(plan(name, wav, 0.0).ranges, words)
        pb = D.score_pauses(pauses(name, wav), words)
        r[name] = (c + dr, pb); tot[name][0] += c + dr; tot[name][1] += pb
    if r["head"][0] > r["main"][0]: worse_pl.append(it)
    if r["head"][0] < r["main"][0]: better_pl += 1
    if r["head"][1] > r["main"][1]: worse_rt.append(it)
    if r["head"][1] < r["main"][1]: better_rt += 1
print(f"advword seed {seed}: {n} pauses; plan cut+dropped main {tot['main'][0]} head {tot['head'][0]} (head worse {len(worse_pl)} {worse_pl[:5]}, better {better_pl}); retime words with pause main {tot['main'][1]} head {tot['head'][1]} (head worse {len(worse_rt)} {worse_rt[:5]}, better {better_rt})")
