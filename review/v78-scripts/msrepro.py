# Usage: PARAKEET_VAD_MIN_SILENCE_MS=<ms> python -I msrepro.py <tree>
# A halting quiet speaker in a 12 s pause between two loud turns; env var path (no monkeypatch).
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from parakeet_service import chunker, retime
assert chunker.__file__.startswith(sys.argv[1])
SR = 16000
def db(x): return 10 ** (x / 20)
rng = np.random.default_rng(0)
P = 12.0; total = 80 + P
wav = rng.standard_normal(int(total * SR)) * db(-55)
wav[: 40 * SR] += rng.standard_normal(40 * SR) * db(-20)
wav[int((40 + P) * SR):] += rng.standard_normal(int(40 * SR)) * db(-20)
words = []
t = 41.0
for i in range(10):
    a, n = int(t * SR), int(0.35 * SR)
    wav[a:a + n] += rng.standard_normal(n) * db(-42); words.append((t, t + 0.35)); t += 0.35 + 0.5
wav = wav.astype(np.float32)
plan = chunker.plan_chunks(wav, target_sec=60.0, max_sec=75.0, context_sec=5.0)
rs = [(a / SR, b / SR) for a, b in plan.ranges]
kept = sum(1 for w0, w1 in words if any(a <= w0 and w1 <= b for a, b in rs))
ps = [(round(a, 2), round(b, 2)) for a, b in retime.pauses(wav) if b > 40 and a < 40 + P]
print(f"VAD_MIN_SILENCE_MS={chunker.VAD_MIN_SILENCE_MS} {sys.argv[1].rsplit('/',1)[1]}: ranges {[(round(a,2), round(b,2)) for a,b in rs]}; words decoded {kept}/10; retime pauses in the gap {ps}")
