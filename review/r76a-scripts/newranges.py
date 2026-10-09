import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/r76a-scripts")
import random
from collections import Counter
from fuzz76 import *
init(); import fuzz76; M = fuzz76.MODS
for cname in ("v2 ctx5", "v2 ctx0", "v3 ctx5"):
  cfg = CONFIGS[cname]
  for gen in ("a", "b"):
    for scale in (1.0, 1.6):
        G = {"a": layout_a, "b": layout_b}[gen]
        c = Counter(); worst = None
        for i in range(12000):
            rng = random.Random(f"r76a|R76A-seed|{cname}|{gen}|{scale}|{i}")
            segs, total = G(rng, scale)
            if total <= cfg["mx"]: continue
            rm = run(M["main"], segs, total, **cfg)[0]; rh = run(M["head"], segs, total, **cfg)[0]
            c["lt5 main"] += sum(b - a < 5 * SR for a, b in rm); c["lt5 head"] += sum(b - a < 5 * SR for a, b in rh)
            c["lt3 main"] += sum(b - a < 3 * SR for a, b in rm); c["lt3 head"] += sum(b - a < 3 * SR for a, b in rh)
            if rm == rh: continue
            new = [r for r in rh if r not in rm]
            if new:
                m = min(new, key=lambda r: r[1] - r[0])
                if worst is None or m[1] - m[0] < worst[0]: worst = (m[1] - m[0], segs, total, sec(rm), sec(rh))
            c["layouts more <5s"] += sum(b - a < 5 * SR for a, b in rh) > sum(b - a < 5 * SR for a, b in rm)
            c["layouts fewer <5s"] += sum(b - a < 5 * SR for a, b in rh) < sum(b - a < 5 * SR for a, b in rm)
        print(cname, gen, scale, dict(c), "shortest new head range:", round(worst[0] / SR, 3) if worst else None)
        if worst and worst[0] < 5 * SR: print("    ", worst[1:])
