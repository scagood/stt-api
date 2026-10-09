import sys, json
root = sys.argv[1]
sys.path.insert(0, root)
import numpy as np
from parakeet_service import chunker
from cmp import metrics, load
SR = 16000
cfg = (25, 30, 20, 5)
own = 20 * SR
orig = chunker._split_oversized
rec = []
def wrap(s, e, m):
    rec.append((s, e)); return orig(s, e, m)
chunker._split_oversized = wrap
def ceil(a, b): return -(-a // b)
from collections import Counter
C = Counter()
ex = {}
for seed in (1, 2, 3):
    lays = json.load(open(f'lay{seed}.json'))
    ra, rb = load('rv74', seed), load('rv74-base', seed)
    for i, (segs, total) in enumerate(lays):
        ma = metrics(segs, total, *ra['v2c5'][i], cfg); mb = metrics(segs, total, *rb['v2c5'][i], cfg)
        if ma['cuts'] <= mb['cuts']: continue
        seg = [(int(round(a * SR)), int(round(b * SR))) for a, b in segs]
        rec.clear()
        chunker._speech_segments = lambda _w: seg
        chunker.plan_chunks(np.zeros(int(round(total*SR)), dtype=np.float32), target_sec=25, max_sec=30, min_sec=20, context_sec=5)
        kind = 'grid'
        for s, e in rec:
            ov = [(max(s, a), min(e, b)) for a, b in seg if min(e, b) > max(s, a)]
            s0, e0 = ov[0][0], ov[-1][1]
            n = ceil(e - s, own)
            if ceil(e0 - s0, own) < n:
                lead = ceil(e - s0, own) < n
                trail = ceil(e0 - s, own) < n
                kind = 'lead' if lead and not trail else ('trail' if trail and not lead else 'both')
                kind += '-first' if s == rec[0][0] else '-later'
                break
        C[kind] += 1
        ex.setdefault(kind, (seed, i, segs, total))
print(C)
for k, v in ex.items(): print(k, v[0], v[1], v[3], v[2][:6])
