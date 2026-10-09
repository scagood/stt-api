import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/rr74-scripts")
import random
from load import *
from fuzz import layout, CONFIGS
new = MODS["c032ee4"]; old = MODS["55df37c"]; prev = MODS["74d1902"]
for cname in ("v2 ctx5", "v2 ctx0", "whisper", "v3 ctx5"):
    cfg = CONFIGS[cname]
    for seed in range(1, 11):
        rng = random.Random(seed * 100 + list(CONFIGS).index(cname))
        for i in range(4000):
            segs, total = layout(rng)
            if total <= cfg["mx"]: continue
            rn, wn, sg, _ = run(new, segs, total, **cfg)
            rp, wp, _, _ = run(prev, segs, total, **cfg)
            ro = run(old, segs, total, **cfg)[0]
            a, p, o = in_speech_cuts(rn, sg), in_speech_cuts(rp, sg), in_speech_cuts(ro, sg)
            if a > p or len(rn) > len(rp) or silent_ranges(rn, sg) > silent_ranges(rp, sg):
                print(cname, seed, i, "segs", segs, "total", total)
                print("   74d1902", show(rp), sec(rp), p)
                print("   c032ee4", show(rn), sec(rn), a)
                print("   55df37c", show(ro), sec(ro), o)
