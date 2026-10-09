"""Tail margin: a quiet word (0.25 s vowel at -42 dBFS) whose fricative tail (0.2 s) is FRIC dBFS,
room tone -65. How much of the word (vowel+tail) is outside head's last range, where main has it all?"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from synth import MAIN, PR, SR, synth
fric = float(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
cut = []; whole = 0
for seed in range(N):
    rng = np.random.default_rng(seed)
    L = rng.uniform(36.0, 39.5); lead = 0.5
    total = int((lead + L + 3.0) * SR)
    seg = (int(lead * SR), int((lead + L) * SR))
    wav, _ = synth([seg], total, rng, 20.0, (50, 350))
    at = int((lead + L + rng.uniform(1.0, 2.2)) * SR)
    v = int(0.25 * SR); f = int(0.2 * SR)
    wav[at:at + v] = rng.standard_normal(v) * 10 ** (-42 / 20)
    wav[at + v:at + v + f] = rng.standard_normal(f) * 10 ** (fric / 20) * np.linspace(1, 0.3, f)
    a, b = at, at + v + f
    pm = MAIN.plan_chunks(wav, target_sec=25.0, max_sec=30.0, min_sec=20.0, context_sec=5.0)
    pp = PR.plan_chunks(wav, target_sec=25.0, max_sec=30.0, min_sec=20.0, context_sec=5.0)
    if not any(r0 <= a and b <= r1 for r0, r1 in pm.ranges):
        continue
    whole += 1
    end = max(w1 for w0, w1 in pp.windows)
    if end < b:
        cut.append((b - max(end, a)) / SR)
print(f"fricative at {fric} dBFS: word whole in main's range in {whole}/{N}; head's last window ends inside the word in {len(cut)} (cut off: max {max(cut, default=0)*1000:.0f} ms, fully lost {sum(c >= 0.45 - 1e-3 for c in cut)})")
