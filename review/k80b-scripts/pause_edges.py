"""Cuts in VAD pauses between meeting ranges: how close to the speech either
side (the padded VAD edge) main and the PR put them."""
import random, zlib
from collections import Counter
import bisect
from h import *
from fuzz80 import layout


def cheap(rng, segs, total):
    env = np.full(total, db(-65), np.float32)
    for a, b in segs:
        t = a
        while t < b:
            w = int(rng.uniform(0.12, 0.6) * SR); g = int(rng.uniform(0.0, 0.35) * SR)
            env[t:min(b, t + w)] = db(rng.uniform(-26, -14))
            env[min(b, t + w):min(b, t + w + g)] = db(rng.uniform(-50, -28))
            t += w + g
    return (np.random.default_rng(rng.integers(1 << 30)).standard_normal(total).astype(np.float32) * env)


N = int(sys.argv[1]); scale = float(sys.argv[2])
for cname, cfg in CONFIGS.items():
    rng = random.Random(f"pe|{cname}|{scale}")
    c = Counter()
    for i in range(N):
        segs_s, total_s = layout(rng, scale)
        if total_s <= cfg["mx"]: continue
        segs = [(int(a * SR), int(b * SR)) for a, b in segs_s]; total = int(total_s * SR)
        wav = cheap(np.random.default_rng(zlib.crc32(f"{cname}|{scale}|{i}".encode())), segs, total)
        ends = [b for a, b in segs]; starts = [a for a, b in segs]
        for k in ("main", "pr"):
            r, w = plan(M[k], wav, segs, cfg)
            for (a, b), (cc, d) in zip(r, r[1:]):
                if b != cc or any(s < b < e for s, e in segs):
                    continue
                j = bisect.bisect_right(ends, b) - 1  # pause after segment j
                if j < 0 or j + 1 >= len(segs): continue
                gap = min(b - ends[j], starts[j + 1] - b) / SR
                c[f"{k}_pause_cuts"] += 1
                c[f"{k}_at_edge(<20ms)"] += gap < 0.02
                c[f"{k}_<100ms"] += gap < 0.1
    print(cname, scale, dict(c))
