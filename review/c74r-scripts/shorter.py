import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/c74r-scripts")
import random
from collections import Counter
from load import *
from fuzz import layout, CONFIGS
new = MODS["c73d2b9"]; old = MODS["main308"]
scale = float(sys.argv[1]); seeds = [int(x) for x in sys.argv[2].split(",")]
for cname in ("v2 ctx0", "whisper", "v2 ctx5"):
    cfg = CONFIGS[cname]; c = Counter(); shown = 0
    for seed in seeds:
        rng = random.Random(seed * 100 + list(CONFIGS).index(cname))
        for i in range(3000):
            segs, total = layout(rng, scale)
            if total <= cfg["mx"]: continue
            rn, _, sg, _ = run(new, segs, total, **cfg); ro = run(old, segs, total, **cfg)[0]
            mn = min(b - a for a, b in rn); mo = min(b - a for a, b in ro)
            if mn < mo - 0.01 * SR:
                c["shorter"] += 1
                if mn < 5 * SR <= mo: c["newly<5s"] += 1
                if mn < 2 * SR <= mo: c["newly<2s"] += 1
                if shown < 2 or (mn < 5 * SR <= mo and shown < 4):
                    shown += 1
                    print(cname, seed, i, segs, total); print("   main", sec(ro), show(ro)); print("   PR  ", sec(rn), show(rn))
    print(cname, dict(c))
