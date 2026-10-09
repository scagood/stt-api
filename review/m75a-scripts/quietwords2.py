import sys
sys.path.insert(0, ".")
sys.path.insert(0, sys.argv[1])
import numpy as np
import quietwords as q  # runs its default report too; fine
from parakeet_service import chunker, retime
SR = chunker.TARGET_SR
print("--- stress / retime detail")
for label, kw in [("default", {}), ("3-5 words, gaps 0.6-1.4", dict(nwords=(3, 5), gap_lo=0.6, gap_hi=1.4)),
                  ("5-7 words, gaps 0.6-1.2", dict(nwords=(5, 7), gap_lo=0.6, gap_hi=1.2))]:
    for level, tone in [(-40, -55), (-46, -70)]:
        tot = lost = in_pause = 0; worst = 0.0
        for seed in range(20):
            wav, words, phrases, (qa, qb) = q.build(seed, level, tone, **kw)
            p = chunker.plan_chunks(wav, target_sec=60.0, max_sec=75.0, context_sec=5.0)
            r = sorted((a / SR, b / SR) for a, b in p.ranges)
            prev = None
            for a, b in r:
                if b <= qa or a >= qb: continue
                if prev is not None and a > prev: worst = max(worst, a - prev)
                prev = b if prev is None else max(prev, b)
            ps = retime.pauses(wav)
            for a, b in words:
                tot += 1
                if q.covered(r, a, b) < (b - a) * 0.99: lost += 1
                m = (a + b) / 2
                if any(x <= m <= y for x, y in ps): in_pause += 1
        print(f"{label} level {level}/{tone}: words {tot}, not decoded {lost}, inside a retime pause {in_pause}, longest cut inside turn {worst:.2f}s")
