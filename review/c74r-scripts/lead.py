import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/c74r-scripts")
import random
from collections import Counter
from load import *
from fuzz import layout, CONFIGS
new = MODS["c73d2b9"]
orig_split = new._split_oversized
calls = []
new._split_oversized = lambda s, e, m: (calls.append((s, e)), orig_split(s, e, m))[1]
for cname in ("v2 ctx5", "v2 ctx0", "whisper", "v3 ctx5"):
    cfg = CONFIGS[cname]; own = int((cfg["mx"] - 2 * cfg["ctx"]) * SR)
    c = Counter(); ex = {}
    for seed, scale in ((1, 1.0), (2, 1.0), (3, 1.0), (4, 1.6)):
        rng = random.Random(seed * 100 + list(CONFIGS).index(cname))
        for i in range(3000):
            segs, total = layout(rng, scale)
            if total <= cfg["mx"]: continue
            calls.clear()
            rn, _, sg, _ = run(new, segs, total, **cfg)
            c["n"] += 1
            for j, (s, e) in enumerate(calls):
                inside = [(max(s, x), min(e, y)) for x, y in sg if x < e and y > s]
                if not inside: c["silent packed"] += 1; continue
                span = inside[-1][1] - inside[0][0]
                need = -(-span // own); got = -(-(e - s) // own)
                if got > need:
                    lead = inside[0][0] - s; trail = e - inside[-1][1]
                    no_lead = -(-(e - s - lead) // own) == need
                    no_trail = -(-(e - s - trail) // own) == need
                    first = j == 0
                    nph = len(inside)
                    if no_lead and not no_trail: kind = ("first-range lead, %s phrase" % ("1" if nph == 1 else ">1")) if first else "lead after cut"
                    elif no_trail and not no_lead: kind = "trail"
                    else: kind = "lead+trail together"
                    c["extra-piece " + kind] += 1
                    if kind not in ex or (e - s) < ex[kind][0]: ex[kind] = (e - s, segs, total, sec([(s, e)]), round(lead / SR, 2), round(trail / SR, 2))
    print(cname, dict(c))
    for k, v in ex.items(): print("   ex", k, v[1:])
