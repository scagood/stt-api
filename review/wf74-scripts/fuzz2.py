import sys, random; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/wf74-scripts")
from load import *
from fuzz import layout, CONFIGS
cfg = CONFIGS["v2 ctx5"]
for seed in (1, 2, 3):
    rng = random.Random(seed)
    st = dict(n=0, a74_gt_55=0, fix_gt_55=0, fix_lt_55=0, a74_lt_55=0, fix_eq_55_when_74_worse=0, worse_with_55_sliver=0)
    for i in range(4000):
        segs, total = layout(rng)
        if total <= cfg["mx"]: continue
        st["n"] += 1
        r = {k: run(m, segs, total, **cfg) for k, m in MODS.items()}
        c = {k: in_speech_cuts(p.ranges, sg) for k, (p, sg, t) in r.items()}
        if c["74"] > c["55"]:
            st["a74_gt_55"] += 1
            if c["fix"] <= c["55"]: st["fix_eq_55_when_74_worse"] += 1
            rr = r["55"][0].ranges
            if c["fix"] > c["55"] and any(e - s < 2 * SR for s, e in rr): st["worse_with_55_sliver"] += 1
        if c["74"] < c["55"]: st["a74_lt_55"] += 1
        if c["fix"] > c["55"]: st["fix_gt_55"] += 1
        if c["fix"] < c["55"]: st["fix_lt_55"] += 1
    print(seed, st)
