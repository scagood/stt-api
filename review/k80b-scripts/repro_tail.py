"""usage: python repro_tail.py WORKTREE
34 s of speech-like audio (3-37 s, -20 dBFS words, 50-350 ms gaps), then a
quiet last word ("bye", -42 dBFS) at 38.3-38.7 s, file ends at 40 s. Real
energy VAD (PARAKEET_VAD=volume default), parakeet-v2 bounds (25/30 s, 5 s
context). Prints VAD, ranges, windows and whether the quiet word is decoded."""
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from parakeet_service import chunker
SR = 16000
db = lambda x: 10.0 ** (x / 20.0)


def word(rng, n, level):
    t = np.arange(n) / SR; f0 = rng.uniform(90, 220)
    s = sum(np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.28)) / h for h in range(1, 12)) + 0.5 * rng.standard_normal(n)
    s *= 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(3, 7) * t + rng.uniform(0, 6.28))
    r = min(int(0.015 * SR), n // 2); s[:r] *= np.linspace(0, 1, r); s[n - r:] *= np.linspace(1, 0, r)
    return s / np.sqrt(np.mean(s ** 2)) * level


seed = int(sys.argv[2]) if len(sys.argv) > 2 else 17
rng = np.random.default_rng(seed)
total, a, b = 40 * SR, 3 * SR, 37 * SR
wav = rng.standard_normal(total) * db(-60)
t = a
while t < b:
    e = min(b, t + int(rng.uniform(0.15, 0.5) * SR))
    if e - t < int(0.05 * SR): break
    wav[t:e] += word(rng, e - t, db(-20))
    g = min(b, e + int(rng.uniform(0.05, 0.35) * SR))
    if g > e: wav[e:g] += rng.standard_normal(g - e) * db(-20) * db(rng.uniform(-30, -10))
    t = g
qa, qb = int(38.3 * SR), int(38.7 * SR)
wav[qa:qb] += word(rng, qb - qa, db(-42))
wav = wav.astype(np.float32)
plan = chunker.plan_chunks(wav, target_sec=25.0, max_sec=30.0, min_sec=20.0, context_sec=5.0)
f = lambda r: [(round(x / SR, 2), round(y / SR, 2)) for x, y in r]
print("VAD     ", f(plan.speech))
print("ranges  ", f(plan.ranges))
print("windows ", f(plan.windows))
print("quiet word 38.30-38.70 s decoded:", any(x <= qa and qb <= y for x, y in plan.windows), "| any of it:", any(x < qb and qa < y for x, y in plan.windows))
