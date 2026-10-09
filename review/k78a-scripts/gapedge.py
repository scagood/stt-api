# Two breaths alone in a 10 s pause: at which spacing do they join into a decoded sound (PR head)?
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/k78a-scripts")
from lib import *
rng = np.random.default_rng(5)
for d in (0.3, 0.45):
    for gap in (0.36, 0.38, 0.39, 0.40, 0.41, 0.42, 0.44, 0.46):
        hits = 0; N = 40
        for i in range(N):
            t0 = 44.0 + float(rng.uniform(0, 0.02))
            wav = build(10, [(t0, d, -42), (t0 + d + gap, d, -42)], seed=100 + i)
            r = plan("head", wav, 0.0).ranges
            if any(b / SR > 40.5 and a / SR < 49.5 for a, b in r): hits += 1
        print(f"breaths {d}s, {gap:.2f}s apart: decoded in {hits}/{N}")
