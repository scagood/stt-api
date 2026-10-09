"""Quiet speaker: phrases 0.6-1.2 s, short words 0.12-0.32 s alone between them (dips >= 0.45 s),
`down` dB under -20 dBFS turns, room tone `tone`. Reports: quiet-turn coverage by plan ranges,
short words lost, longest uncovered stretch inside the quiet turn; retime pauses covering words."""
import sys
sys.path.insert(0, ".")
import numpy as np
from parakeet_service import chunker, retime
chunker.VAD = "volume"; chunker.VAD_GATE_DB = None
SR = chunker.TARGET_SR
db = lambda x: 10 ** (x / 20)

def speech(sec, level, rng):
    # syllabic: 120-250 ms syllables with 20-60 ms dips 15 dB down
    out = []
    n = int(sec * SR)
    while sum(map(len, out)) < n:
        out.append(rng.standard_normal(int(rng.uniform(0.12, 0.25) * SR)) * db(level))
        out.append(rng.standard_normal(int(rng.uniform(0.02, 0.06) * SR)) * db(level - 15))
    return np.concatenate(out)[:n]

def build(seed, level, tone, gap_lo=0.45, gap_hi=0.9, nwords=(1, 3), turn_sec=15.0):
    rng = np.random.default_rng(seed)
    pieces, words, phrases = [], [], []
    t = 0.0
    while t < turn_sec:
        d = rng.uniform(0.6, 1.2)
        pieces.append((t, speech(d, level, rng))); phrases.append((t, t + d)); t += d
        for _ in range(rng.integers(nwords[0], nwords[1] + 1)):
            t += rng.uniform(gap_lo, gap_hi)
            d = rng.uniform(0.12, 0.32)
            pieces.append((t, rng.standard_normal(int(d * SR)) * db(level))); words.append((t, t + d)); t += d
        t += rng.uniform(gap_lo, gap_hi)
    quiet_len = t
    total = 40 + quiet_len + 40
    wav = rng.standard_normal(int(total * SR)) * db(tone)
    wav[: 40 * SR] += rng.standard_normal(40 * SR) * db(-20)
    off = 40 + quiet_len
    wav[int(off * SR): int(off * SR) + 40 * SR] += rng.standard_normal(40 * SR) * db(-20)[()] if False else rng.standard_normal(min(40 * SR, wav.size - int(off * SR))) * db(-20)
    for at, p in pieces:
        a = int((40 + at) * SR); wav[a: a + p.size] += p
    sh = lambda xs: [(a + 40, b + 40) for a, b in xs]
    return wav.astype(np.float32), sh(words), sh(phrases), (40.0, off)

def covered(ranges, a, b):
    return sum(max(0, min(y, b) - max(x, a)) for x, y in ranges)

for level, tone in [(-40, -55), (-46, -70), (-44, -70)]:
    lost_w = lost_p = tot_w = 0; worst_gap = 0.0; cov = []; rt_bad = 0
    for seed in range(20):
        wav, words, phrases, (qa, qb) = build(seed, level, tone)
        p = chunker.plan_chunks(wav, target_sec=60.0, max_sec=75.0, context_sec=5.0)
        r = [(a / SR, b / SR) for a, b in p.ranges]
        cov.append(covered(r, qa, qb) / (qb - qa))
        for a, b in words:
            tot_w += 1
            if covered(r, a, b) < (b - a) * 0.99: lost_w += 1
        for a, b in phrases:
            if covered(r, a, b) < (b - a) * 0.99: lost_p += 1
        # uncovered stretches strictly inside the quiet turn
        edges = sorted(r)
        prev = qa
        for a, b in edges:
            if b <= qa or a >= qb: continue
            if a > prev: worst_gap = max(worst_gap, a - prev) if prev > qa else worst_gap
            prev = max(prev, b)
        ps = retime.pauses(wav)
        for a, b in words:
            m = (a + b) / 2
            if any(x <= m <= y and y - x >= 3.0 for x, y in ps): rt_bad += 1
    print(f"level {level} tone {tone}: words {tot_w}, words not covered {lost_w}, phrases not covered {lost_p}, "
          f"quiet-turn coverage min {min(cov):.3f}, longest cut inside turn {worst_gap:.2f}s, words inside >=3s retime pauses {rt_bad}")
