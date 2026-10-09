"""Sensitivity to PARAKEET_VAD_MIN_SILENCE_MS (now also the join dip inside loud_frames)."""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, sys.argv[1])
import io, contextlib
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    import repro
from parakeet_service import chunker, retime
SR = chunker.TARGET_SR
db = lambda x: 10 ** (x / 20)

def realistic_quiet(seed, level, tone, turn_sec=15.0):
    # words 0.15-0.45 s, gaps between words 0.05-0.25 s, phrases of 3-8 words, 0.4-1.0 s between phrases
    rng = np.random.default_rng(seed)
    segs = []; t = 0.0
    while t < turn_sec:
        for _ in range(rng.integers(3, 9)):
            d = rng.uniform(0.15, 0.45); segs.append((t, d)); t += d + rng.uniform(0.05, 0.25)
        t += rng.uniform(0.4, 1.0)
    total = 80 + t
    wav = rng.standard_normal(int(total * SR)) * db(tone)
    wav[: 40 * SR] += rng.standard_normal(40 * SR) * db(-20)
    off = int((40 + t) * SR)
    wav[off:] += rng.standard_normal(wav.size - off) * db(-20)
    for a, d in segs:
        i = int((40 + a) * SR); n = int(d * SR); wav[i:i + n] += rng.standard_normal(n) * db(level)
    return wav.astype(np.float32), (40.0, 40 + t), [(40 + a, 40 + a + d) for a, d in segs]

for ms in (400, 200, 100, 1000, 3000, 5000):
    chunker.VAD_MIN_SILENCE_MS = ms
    cov = []; lostw = 0; totw = 0
    for seed in range(10):
        wav, (qa, qb), words = realistic_quiet(seed, -44, -70)
        r = [(a / SR, b / SR) for a, b in chunker.plan_chunks(wav, target_sec=60.0, max_sec=75.0, context_sec=5.0).ranges]
        cov.append(sum(max(0, min(b, qb) - max(a, qa)) for a, b in r) / (qb - qa))
        for a, b in words:
            totw += 1; lostw += sum(max(0, min(y, b) - max(x, a)) for x, y in r) < 0.99 * (b - a)
    a3 = repro.plan(repro.build(15, [(43, 0.25, -42), (48, 0.25, -42), (53, 0.25, -42)]), 0)[0]
    b3 = repro.plan(repro.build(30, [(42 + 4 * i, 0.2, -45) for i in range(7)]), 5)[0]
    c3 = repro.plan(repro.build(10, [(43.0, 0.3, -42), (46.0, 0.3, -42)]), 0)[0]
    print(f"min_silence {ms} ms: quiet turn coverage min {min(cov):.2f} mean {np.mean(cov):.2f}, words lost {lostw}/{totw}; breaths (a) {a3}; (b) {b3}; (c') {c3}")
