import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/r76a-scripts")
import random, json, os
from collections import Counter
from multiprocessing import Pool
from load76 import *
CONFIGS = {
  "v2 ctx5": dict(target=25.0, mx=30.0, ctx=5.0, mn=20.0),
  "v2 ctx0": dict(target=25.0, mx=30.0, ctx=0.0, mn=20.0),
  "v3 ctx5": dict(target=60.0, mx=75.0, ctx=5.0, mn=20.0),
  "whisper": dict(target=25.0, mx=30.0, ctx=0.0, mn=20.0),
}
EXTRA = {
  "v3 ctx0": dict(target=60.0, mx=75.0, ctx=0.0, mn=20.0),
  "v2 ctx3 min10": dict(target=25.0, mx=30.0, ctx=3.0, mn=10.0),
  "v2 ctx5 min0": dict(target=25.0, mx=30.0, ctx=5.0, mn=0.0),
}
def layout_a(rng, scale):  # same shape as c74r fuzz.layout
    t = rng.uniform(0, 4); segs = []
    for _ in range(rng.randint(1, 8)):
        L = rng.choice([rng.uniform(0.3, 6), rng.uniform(5, 25), rng.uniform(15, 50)]) * scale
        segs.append((round(t, 3), round(t + L, 3)))
        t += L + rng.choice([rng.uniform(0.05, 1.0), rng.uniform(0.5, 2.9), rng.uniform(3, 8)])
    total = round(segs[-1][1] + rng.choice([0, rng.uniform(0, 1.5), rng.uniform(1, 6)]), 3)
    return segs, total
def layout_b(rng, scale):  # denser: many phrases, pauses mostly under trim_gap, near-boundary lengths
    t = rng.choice([0, rng.uniform(0, 1), rng.uniform(1, 6)]); segs = []
    for _ in range(rng.randint(2, 20)):
        L = rng.choice([rng.uniform(0.2, 3), rng.uniform(3, 12), rng.uniform(12, 22), rng.uniform(18, 21), rng.uniform(19.5, 20.5), rng.uniform(25, 70)]) * scale
        segs.append((round(t, 3), round(t + L, 3)))
        t += L + rng.choice([rng.uniform(0.4, 1.2), rng.uniform(1, 3.0), rng.uniform(2.8, 3.2), rng.uniform(3, 12)])
    total = round(segs[-1][1] + rng.choice([0, rng.uniform(0, 3), rng.uniform(2.5, 10)]), 3)
    return segs, total
def metrics(r, w, segs, total, cfg):
    own = max(1, int(cfg["mx"] * SR) - 2 * int(cfg["ctx"] * SR)); mx = int(cfg["mx"] * SR)
    bad = []
    if any(not (0 <= a < b <= total) for a, b in r): bad.append("outside")
    if any(b > c for (a, b), (c, d) in zip(r, r[1:])): bad.append("overlap")
    if any(b - a > own for a, b in r): bad.append("range>own")
    if len(w) != len(r): bad.append("nwin")
    if any(b - a > mx for a, b in w): bad.append("window>max")
    if any(not (0 <= wa <= a and b <= wb <= total) for (a, b), (wa, wb) in zip(r, w)): bad.append("window!>=range")
    if any(type(x) is not int for p in r + w for x in p): bad.append("nonint")
    return dict(n=len(r), isc=in_speech_cuts(r, segs), silent=silent_ranges(r, segs),
                short=sum(1 for a, b in r if b - a < 2 * SR), lost=speech_lost(r, segs), bad=bad)
MODS = None
def init():
    global MODS
    MODS = {"main": load("main"), "head": load("head")}
def work(args):
    cname, cfg, gen, scale, seedtag, lo, hi = args
    G = {"a": layout_a, "b": layout_b}[gen]
    tot = {k: Counter() for k in MODS}; cmp = Counter(); ex = {}
    for i in range(lo, hi):
        rng = random.Random(f"r76a|{seedtag}|{cname}|{gen}|{scale}|{i}")
        segs, total = G(rng, scale)
        if total <= cfg["mx"]: continue
        m = {}
        for k, mod in MODS.items():
            r, w, sg, t = run(mod, segs, total, **cfg)
            m[k] = metrics(r, w, sg, t, cfg); m[k]["r"] = r
        for k in MODS:
            x = m[k]; tt = tot[k]
            tt["layouts"] += 1
            for key in ("n", "isc", "silent", "short", "lost"): tt[key] += x[key]
            tt["lost_layouts"] += x["lost"] > 0; tt["bad_layouts"] += bool(x["bad"])
            for b in x["bad"]: tt["bad:" + b] += 1
        a, b = m["head"], m["main"]
        for key in ("n", "isc", "silent", "short", "lost"):
            if a[key] > b[key]:
                cmp[key + "_worse"] += 1
                ex.setdefault(key, (segs, total, sec(b["r"]), sec(a["r"])))
            elif a[key] < b[key]: cmp[key + "_better"] += 1
        if a["r"] != b["r"]: cmp["changed"] += 1
        if a["bad"]: ex.setdefault("bad", (segs, total, a["bad"], sec(a["r"])))
    return cname, gen, scale, {k: dict(v) for k, v in tot.items()}, dict(cmp), ex
if __name__ == "__main__":
    N = int(sys.argv[1]); seedtag = sys.argv[2]; which = sys.argv[3] if len(sys.argv) > 3 else "main"
    gens = sys.argv[4].split(",") if len(sys.argv) > 4 else ["a"]
    cfgs = CONFIGS if which == "main" else EXTRA
    jobs = []; step = 2000
    for cname, cfg in cfgs.items():
        for gen in gens:
            for scale in (1.0, 1.6):
                for lo in range(0, N, step): jobs.append((cname, cfg, gen, scale, seedtag, lo, min(N, lo + step)))
    agg = {}
    with Pool(os.cpu_count(), initializer=init) as p:
        for cname, gen, scale, tot, cmp, ex in p.imap_unordered(work, jobs):
            key = (cname, gen, scale)
            A = agg.setdefault(key, {"tot": {k: Counter() for k in ("main", "head")}, "cmp": Counter(), "ex": {}})
            for k in tot: A["tot"][k].update(tot[k])
            A["cmp"].update(cmp)
            for kk, v in ex.items(): A["ex"].setdefault(kk, v)
    for key in sorted(agg):
        A = agg[key]
        print(f"== {key[0]} gen={key[1]} scale={key[2]}")
        for k in ("main", "head"): print(f"   {k:5s}", dict(A["tot"][k]))
        print("   head vs main", dict(sorted(A["cmp"].items())))
        for kk, v in A["ex"].items(): print("   ex", kk, v)
