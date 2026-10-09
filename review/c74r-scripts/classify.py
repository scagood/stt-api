import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/c74r-scripts")
import random
from collections import Counter
from load import *
from fuzz import layout, CONFIGS
new = MODS["c73d2b9"]; old = MODS["main308"]
def recorder(mod):
    orig = mod._split_oversized; calls = []
    def rec(*a):
        calls.append((a[0], a[1])); return orig(*a)
    mod._split_oversized = rec
    return calls
cn = recorder(new); co = recorder(old)
scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
seeds = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 3]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
for cname in ("v2 ctx5", "v2 ctx0", "whisper", "v3 ctx5"):
    cfg = CONFIGS[cname]; own = int((cfg["mx"] - 2 * cfg["ctx"]) * SR)
    c = Counter(); exs = {}
    for seed in seeds:
        rng = random.Random(seed * 100 + list(CONFIGS).index(cname))
        for i in range(N):
            segs, total = layout(rng, scale)
            if total <= cfg["mx"]: continue
            cn.clear(); co.clear()
            rn, _, sg, _ = run(new, segs, total, **cfg); pn = [x for x in cn if True]
            # old module calls _split_oversized inside the loop too (last_piece); keep only final ones = last len(packed) calls
            ro, _, _, _ = run(old, segs, total, **cfg)
            a, b = in_speech_cuts(rn, sg), in_speech_cuts(ro, sg)
            c["layouts"] += 1
            if len(rn) > len(ro): c["n_worse"] += 1
            if silent_ranges(rn, sg) > silent_ranges(ro, sg): c["silent_worse"] += 1
            if a <= b: continue
            c["isc_worse"] += 1
            # PR packed ranges: the calls from the output loop are the final len(...) calls
            packed_new = pn
            # does the PR split a packed range that holds an internal pause, where main's split cut in that pause?
            cuts_old = {x for (s, e), (s2, e2) in zip(ro, ro[1:]) if e == s2 for x in [e]}
            grid = False
            for s, e in packed_new:
                pauses = [(e1, s2) for (s1, e1), (s2, e2) in zip(sg, sg[1:]) if s < e1 and s2 < e]
                if any(any(p0 <= x <= p1 for x in cuts_old) for p0, p1 in pauses):
                    grid = True
                    # would cutting at that pause first then splitting evenly need no more pieces?
                    for p0, p1 in pauses:
                        for x in cuts_old:
                            if p0 <= x <= p1:
                                k = -(-(x - s) // own) + -(-(e - x) // own); kk = -(-(e - s) // own)
                                c["grid: pause-cut free" if k <= kk else "grid: pause-cut costs a piece"] += 1
            same_packed = sorted(set(packed_new)) == sorted({(s, e) for s, e in packed_new})
            key = "grid (main cut in an internal pause the PR's even split misses)" if grid else "other"
            c[key] += 1
            if key not in exs: exs[key] = (segs, total, sec(ro), show(ro), sec(rn), show(rn), b, a)
    print(cname, dict(c))
    for k, v in exs.items(): print("  ex", k, v)
