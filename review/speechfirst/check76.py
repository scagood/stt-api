import os, sys, random, importlib.util
import numpy as np
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, "/home/user/stt-api")
import parakeet_service, parakeet_service.config
SR = parakeet_service.config.TARGET_SR
def load(name):
    spec = importlib.util.spec_from_file_location(f"parakeet_service._c76_{name}", f"{D}/chunker_{name}.py")
    m = importlib.util.module_from_spec(spec); m.__package__ = "parakeet_service"; spec.loader.exec_module(m); return m
M = {"main": load("main"), "new": load("pr76b")}
def layout(rng, scale):
    t = rng.uniform(0, 4); segs = []
    for _ in range(rng.randint(1, 8)):
        L = rng.choice([rng.uniform(0.3, 6), rng.uniform(5, 25), rng.uniform(15, 50)]) * scale
        segs.append((t, t + L)); t += L + rng.choice([rng.uniform(0.05, 1.0), rng.uniform(0.5, 2.9), rng.uniform(3, 8)])
    return segs, segs[-1][1] + rng.choice([0, rng.uniform(0, 1.5), rng.uniform(1, 6)])
def plan(mod, segs_s, total_s, target, mx, ctx, trim):
    segs = [(int(round(a*SR)), int(round(b*SR))) for a, b in segs_s]; total = int(round(total_s*SR))
    mod._speech_segments = lambda _w: segs; mod.CHUNK_TRIM_SILENCE_SEC = trim
    p = mod.plan_chunks(np.zeros(total, np.float32), target_sec=target, max_sec=mx, min_sec=20.0, context_sec=ctx)
    return p.ranges, p.windows, segs, total
def metr(r, w, segs, total, mx, ctx):
    own = int((mx-2*ctx)*SR)
    isc = sum(1 for (a,b),(c,d) in zip(r, r[1:]) if b == c and any(s < b < e for s, e in segs))
    sil = sum(1 for a,b in r if not any(s < b and e > a for s, e in segs))
    short = sum(1 for a,b in r if b-a < 2*SR)
    lost = sum((e-s) - sum(max(0, min(b,e)-max(a,s)) for a,b in r) for s,e in segs)
    bad = any(b-a > own for a,b in r) or any(b-a > int(mx*SR) for a,b in w) or any(not (wa <= a and b <= wb) for (a,b),(wa,wb) in zip(r,w))
    return dict(n=len(r), isc=isc, sil=sil, short=short, lost=lost > 0, bad=bad)
cases = [("v2 ctx5 trim3", 25,30,5,3), ("v2 ctx0 trim3", 25,30,0,3), ("whisper trim3", 25,30,0,3), ("v3 ctx5 trim3", 60,75,5,3),
         ("v2 ctx5 trim1", 25,30,5,1), ("v2 ctx5 trim0.5", 25,30,5,0.5), ("ctx7.5 trim6", 25,30,7.5,6), ("ctx7.5 trim8", 25,30,7.5,8)]
N = int(sys.argv[1])
for name, target, mx, ctx, trim in cases:
    worse = dict(n=0, isc=0, sil=0, short=0, lost=0, bad=0); tot = {"main": 0, "new": 0}; L = 0
    for scale in (1.0, 1.6):
        rng = random.Random(f"x76|{name}|{scale}")
        for _ in range(N):
            segs_s, total_s = layout(rng, scale)
            if total_s <= mx: continue
            L += 1; m = {k: metr(*plan(mod, segs_s, total_s, target, mx, ctx, trim), mx, ctx) for k, mod in M.items()}
            for k in ("n", "isc", "sil", "short"): worse[k] += m["new"][k] > m["main"][k]
            worse["lost"] += m["new"]["lost"]; worse["bad"] += m["new"]["bad"]
            tot["main"] += m["main"]["isc"]; tot["new"] += m["new"]["isc"]
    print(f"{name:16s} layouts={L:6d} in-speech cuts main {tot['main']:6d} -> new {tot['new']:6d} | new worse than main in: {worse}")
