import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth import *
import random
for cname in CONFIGS:
    cfg = CONFIGS[cname]; own = int((cfg["mx"] - 2 * cfg["ctx"]) * SR)
    rng = random.Random(f"mins|{cname}"); nrng = np.random.default_rng(1)
    shorter = quarter = L = 0; worst = 1e9
    for _ in range(300):
        segs_s, total_s = layout(rng, 1.6)
        if total_s <= cfg["mx"]: continue
        total = int(round(total_s * SR)); segs = [(int(round(a * SR)), int(round(b * SR))) for a, b in segs_s]
        wav, words = synth(segs, total, nrng, 20.0, (50, 350))
        rm, _ = plan(MAIN, wav, segs, cfg); rp, _ = plan(PR, wav, segs, cfg)
        L += 1
        mm, mp = min(b - a for a, b in rm), min(b - a for a, b in rp)
        shorter += mp < mm - 1
        quarter += sum(1 for a, b in rp if b - a < own // 4) > sum(1 for a, b in rm if b - a < own // 4)
        worst = min(worst, (mp - mm) / SR)
    print(f"{cname}: layouts {L}, #80 shortest piece shorter than main's in {shorter}, more pieces < max/4 in {quarter}, most shortened {worst:.2f}s", flush=True)
