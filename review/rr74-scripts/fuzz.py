import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/rr74-scripts")
import random
from collections import Counter
from load import *
CONFIGS = {
  "v2 ctx5": dict(target=25.0, mx=30.0, ctx=5.0),
  "v2 ctx0": dict(target=25.0, mx=30.0, ctx=0.0),
  "v3 ctx5": dict(target=60.0, mx=75.0, ctx=5.0),
  "whisper": dict(target=25.0, mx=30.0, ctx=0.0),
}
def layout(rng, scale=1.0):
    t = rng.uniform(0, 4)
    segs = []
    for _ in range(rng.randint(1, 8)):
        L = rng.choice([rng.uniform(0.3, 6), rng.uniform(5, 25), rng.uniform(15, 50)]) * scale
        segs.append((round(t, 3), round(t + L, 3)))
        t += L + rng.choice([rng.uniform(0.05, 1.0), rng.uniform(0.5, 2.9), rng.uniform(3, 8)])
    total = round(segs[-1][1] + rng.choice([0, rng.uniform(0, 1.5), rng.uniform(1, 6)]), 3)
    return segs, total
def metrics(ranges, windows, segs, total, cfg):
    own = int((cfg["mx"] - 2 * cfg["ctx"]) * SR); mx = int(cfg["mx"] * SR)
    bad = []
    if any(not (0 <= a < b <= total) for a, b in ranges): bad.append("outside")
    if any(b > c for (a, b), (c, d) in zip(ranges, ranges[1:])): bad.append("overlap")
    if any(b - a > own for a, b in ranges): bad.append("range>own")
    if any(b - a > mx for a, b in windows): bad.append("window>max")
    if any(not (wa <= a and b <= wb) for (a, b), (wa, wb) in zip(ranges, windows)): bad.append("window!>=range")
    return dict(isc=in_speech_cuts(ranges, segs), silent=silent_ranges(ranges, segs), n=len(ranges),
                minlen=min((b - a for a, b in ranges), default=0), lost=speech_lost(ranges, segs), bad=bad)
if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    NEW = os.environ.get("NEW", "c032ee4")
    for cname, cfg in CONFIGS.items():
        rng = random.Random(seed * 100 + list(CONFIGS).index(cname))
        tot = {k: Counter() for k in NAMES}
        cmp = {k: Counter() for k in NAMES if k != NEW}
        ex = {}
        chunked = 0
        for i in range(N):
            segs, total = layout(rng, scale)
            if total <= cfg["mx"]: continue
            chunked += 1
            m = {}
            for k, mod in MODS.items():
                r, w, sg, t = run(mod, segs, total, **cfg)
                m[k] = metrics(r, w, sg, t, cfg); m[k]["r"] = r
            for k in NAMES:
                x = m[k]
                tot[k]["isc"] += x["isc"]; tot[k]["silent"] += x["silent"]; tot[k]["n"] += x["n"]
                tot[k]["lost_layouts"] += x["lost"] > 0
                tot[k]["bad"] += bool(x["bad"])
                tot[k]["min<2s"] += x["minlen"] < 2 * SR
                tot[k]["min<5s"] += x["minlen"] < 5 * SR
                if x["bad"] and ("bad", k) not in ex: ex[("bad", k)] = (segs, total, x["bad"])
            for k in cmp:
                a, b = m[NEW], m[k]
                for key in ("isc", "silent", "n"):
                    if a[key] > b[key]:
                        cmp[k][key + "_worse"] += 1
                        if (key, k) not in ex: ex[(key, k)] = (segs, total, show(b["r"]), show(a["r"]), b[key], a[key])
                    if a[key] < b[key]: cmp[k][key + "_better"] += 1
                if a["minlen"] < b["minlen"] - 0.01 * SR: cmp[k]["minlen_shorter"] += 1
                if a["minlen"] > b["minlen"] + 0.01 * SR: cmp[k]["minlen_longer"] += 1
        print(f"== {cname}: {chunked} chunked layouts")
        for k in NAMES: print(f"  totals {k:12s}", dict(tot[k]))
        for k in cmp: print(f"  {NEW} vs {k:12s}", dict(sorted(cmp[k].items())))
        for key, v in ex.items():
            if key[1] in os.environ.get("EX", "74d1902").split(",") or key[0] == "bad": print("   ex", key, v)
