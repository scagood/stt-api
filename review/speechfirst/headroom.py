import os, sys, random, math, importlib.util
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/user/stt-api")
import parakeet_service, parakeet_service.config
SR = parakeet_service.config.TARGET_SR
def load(name):
    spec = importlib.util.spec_from_file_location(f"parakeet_service._sf_{name}", f"{D}/chunker_{name}.py")
    m = importlib.util.module_from_spec(spec); m.__package__ = "parakeet_service"; spec.loader.exec_module(m); return m
MODS = {n: load(n) for n in ("main", "pr76")}
CONFIGS = {"v2 ctx5": (25., 30., 5.), "v2 ctx0": (25., 30., 0.), "v3 ctx5": (60., 75., 5.), "whisper": (25., 30., 0.)}
def layout(rng, scale=1.0):
    t = rng.uniform(0, 4); segs = []
    for _ in range(rng.randint(1, 8)):
        L = rng.choice([rng.uniform(0.3, 6), rng.uniform(5, 25), rng.uniform(15, 50)]) * scale
        segs.append((t, t + L)); t += L + rng.choice([rng.uniform(0.05, 1.0), rng.uniform(0.5, 2.9), rng.uniform(3, 8)])
    return segs, segs[-1][1] + rng.choice([0, rng.uniform(0, 1.5), rng.uniform(1, 6)])
def plan(mod, segs_s, total_s, target, mx, ctx):
    segs = [(int(round(a*SR)), int(round(b*SR))) for a, b in segs_s]; total = int(round(total_s*SR))
    mod._speech_segments = lambda _w: segs
    return mod.plan_chunks(np.zeros(total, np.float32), target_sec=target, max_sec=mx, min_sec=20.0, context_sec=ctx).ranges, segs
def isc(ranges, segs):
    return sum(1 for (a, b), (c, d) in zip(ranges, ranges[1:]) if b == c and any(s < b < e for s, e in segs))
def best(segs, own):
    # speech-first optimum: fewest in-speech cuts (only phrases longer than own are cut), then fewest pieces
    n = len(segs); INF = 10**9; dp = [0] + [INF]*n
    for j in range(1, n+1):
        for i in range(j, 0, -1):
            span = segs[j-1][1] - segs[i-1][0]
            if i == j: pieces = max(1, math.ceil((segs[i-1][1]-segs[i-1][0]) / own))
            elif span <= own: pieces = 1
            else: break
            dp[j] = min(dp[j], dp[i-1] + pieces)
    lb = sum(max(0, math.ceil((b-a)/own) - 1) for a, b in segs)
    return lb, dp[n]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
for scale in (1.0, 1.6):
  for cname, (target, mx, ctx) in CONFIGS.items():
    rng = random.Random(4242 + int(scale*10) + list(CONFIGS).index(cname))
    own = int((mx - 2*ctx) * SR); acc = dict(n=0, lb=0, best_p=0)
    for k in MODS: acc[k+"_isc"] = 0; acc[k+"_p"] = 0
    for _ in range(N):
        segs_s, total_s = layout(rng, scale)
        if total_s <= mx: continue
        acc["n"] += 1
        for k, m in MODS.items():
            r, segs = plan(m, segs_s, total_s, target, mx, ctx); acc[k+"_isc"] += isc(r, segs); acc[k+"_p"] += len(r)
        lb, bp = best(segs, own); acc["lb"] += lb; acc["best_p"] += bp
    a = acc
    print(f"x{scale} {cname:8s} layouts={a['n']:5d} | in-speech cuts: main {a['main_isc']:5d}  #76 {a['pr76_isc']:5d}  unavoidable {a['lb']:5d}"
          f"  -> avoidable main {a['main_isc']-a['lb']:5d} #76 {a['pr76_isc']-a['lb']:5d} | pieces: main {a['main_p']} #76 {a['pr76_p']} speech-first optimum {a['best_p']}")
