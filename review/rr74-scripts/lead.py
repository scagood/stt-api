import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/rr74-scripts")
import random
from collections import Counter
from load import *
from fuzz import layout, CONFIGS
new = MODS["c032ee4"]; old = MODS["55df37c"]; prev = MODS["74d1902"]
orig_split = new._split_oversized
calls = []
new._split_oversized = lambda s, e, m: (calls.append((s, e)), orig_split(s, e, m))[1]
for cname in ("v2 ctx5", "v2 ctx0", "whisper", "v3 ctx5"):
    cfg = CONFIGS[cname]; own = int((cfg["mx"] - 2 * cfg["ctx"]) * SR)
    c = Counter(); ex = {}
    for seed in (1, 2, 3):
        rng = random.Random(seed * 100 + list(CONFIGS).index(cname))
        for i in range(4000):
            segs, total = layout(rng)
            if total <= cfg["mx"]: continue
            calls.clear()
            rn, _, sg, _ = run(new, segs, total, **cfg)
            packed = list(calls)
            ro = run(old, segs, total, **cfg)[0]
            rp = run(prev, segs, total, **cfg)[0]
            a, b, p = in_speech_cuts(rn, sg), in_speech_cuts(ro, sg), in_speech_cuts(rp, sg)
            c["n"] += 1
            if a > p: c["worse_than_74d1902"] += 1
            if a > b: c["worse_than_55"] += 1
            if a < b: c["better_than_55"] += 1
            # any packed range where silence adds a piece
            for j, (s, e) in enumerate(packed):
                inside = [(max(s, x), min(e, y)) for x, y in sg if x < e and y > s]
                if not inside: continue
                span = inside[-1][1] - inside[0][0]
                need = -(-span // own); got = -(-(e - s) // own)
                if got > need:
                    lead = inside[0][0] - s; trail = e - inside[-1][1]
                    no_lead = -(-(e - s - lead) // own) == need
                    no_trail = -(-(e - s - trail) // own) == need
                    first = j == 0
                    kind = ("first-range lead" if first else "lead after cut") if no_lead and not no_trail else ("trail" if no_trail and not no_lead else "either/both")
                    c["extra-piece " + kind] += 1
                    if a > b: c["worse55 & extra-piece " + kind] += 1
                    if kind not in ex or (e - s) < ex[kind][0]: ex[kind] = (e - s, segs, total, sec([(s, e)]), round(lead / SR, 2), round(trail / SR, 2))
    print(cname, dict(c))
    for k, v in ex.items(): print("   ex", k, v[1:])
