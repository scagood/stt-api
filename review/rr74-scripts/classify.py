import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/rr74-scripts")
import random
from collections import Counter
from load import *
from fuzz import layout, CONFIGS
cname = sys.argv[1] if len(sys.argv) > 1 else "v2 ctx5"
cfg = CONFIGS[cname]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
new = MODS["c032ee4"]; old = MODS["55df37c"]
orig_split = new._split_oversized
calls = []
def rec(s, e, m):
    calls.append((s, e)); return orig_split(s, e, m)
new._split_oversized = rec
own = int((cfg["mx"] - 2 * cfg["ctx"]) * SR)
rng = random.Random(seed * 100 + list(CONFIGS).index(cname))
c = Counter(); exs = {}
for i in range(4000):
    segs, total = layout(rng)
    if total <= cfg["mx"]: continue
    calls.clear()
    rn, _, sg, _ = run(new, segs, total, **cfg)
    packed = list(calls)
    ro, _, _, _ = run(old, segs, total, **cfg)
    a, b = in_speech_cuts(rn, sg), in_speech_cuts(ro, sg)
    if a <= b: continue
    causes = set()
    for s, e in packed:
        inside = [(max(s, x), min(e, y)) for x, y in sg if x < e and y > s]
        if not inside: causes.add("silent packed"); continue
        span = inside[-1][1] - inside[0][0]
        if -(-(e - s) // own) > -(-span // own):
            lead = inside[0][0] - s; trail = e - inside[-1][1]
            causes.add("lead-silence" if lead >= trail else "trail-silence")
    key = "+".join(sorted(causes)) or "grid (no extra piece from silence)"
    # for the lead-silence case: is it the first range (file-start margin) or after a mid-pause cut?
    c[key] += 1
    if key not in exs: exs[key] = (segs, total, show(ro), show(rn), sec(rn), b, a)
print(cname, dict(c))
for k, v in exs.items(): print("  ex", k, v)
