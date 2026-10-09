import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/r76a-scripts")
import random
from collections import Counter
from fuzz76 import *
init(); import fuzz76; M = fuzz76.MODS
cname = sys.argv[1]; gen = sys.argv[2]; scale = float(sys.argv[3]); N = int(sys.argv[4])
cfg = {**CONFIGS, **EXTRA}[cname]; G = {"a": layout_a, "b": layout_b}[gen]
c = Counter(); shown = 0; mins = Counter()
for i in range(N):
    rng = random.Random(f"r76a|R76A-seed|{cname}|{gen}|{scale}|{i}")
    segs, total = G(rng, scale)
    if total <= cfg["mx"]: continue
    out = {k: run(m, segs, total, **cfg) for k, m in M.items()}
    rm, rh = out["main"][0], out["head"][0]; sg = out["main"][2]
    if rm == rh: continue
    im, ih = in_speech_cuts(rm, sg), in_speech_cuts(rh, sg)
    mm, mh = min(b - a for a, b in rm), min(b - a for a, b in rh)
    c["changed"] += 1
    c["minlen shorter" if mh < mm else "minlen longer" if mh > mm else "minlen same"] += 1
    c["<5s main"] += sum(b - a < 5 * SR for a, b in rm); c["<5s head"] += sum(b - a < 5 * SR for a, b in rh)
    c["<10s main"] += sum(b - a < 10 * SR for a, b in rm); c["<10s head"] += sum(b - a < 10 * SR for a, b in rh)
    mins[round(mh / SR)] += 1
    if ih == im and shown < 3:
        shown += 1
        print("same isc", segs, total); print("   main", sec(rm)); print("   head", sec(rh))
print(dict(c)); print("head min range len hist (s) on changed:", sorted(mins.items()))
