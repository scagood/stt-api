import sys
sys.path.insert(0, ".")
sys.path.insert(0, sys.argv[1])
import io, contextlib
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    import repro
from parakeet_service import retime
rng = np.random.default_rng(42)
bad_chunk = bad_rt = 0; N = 200; examples = []
for i in range(N):
    pause = float(rng.uniform(5, 40))
    sounds = []; t = 40.3
    while True:
        t += rng.uniform(0.45, 6.0)
        d = float(rng.uniform(0.08, 0.45))
        if t + d > 40 + pause - 0.3: break
        sounds.append((t, d, float(rng.uniform(-48, -38)))); t += d
    ctx = float(rng.choice([0.0, 5.0]))
    wav = repro.build(pause, sounds, seed=i)
    r, _ = repro.plan(wav, ctx)
    inside = [(a, b) for a, b in r if b > 40.5 and a < 40 + pause - 0.5]
    if inside:
        bad_chunk += 1
        if len(examples) < 3: examples.append((round(pause, 1), len(sounds), r))
    ps = retime.pauses(wav)
    if not any(a <= 40.2 and b >= 40 + pause - 0.2 for a, b in ps):
        bad_rt += 1
print(f"{N} breath-only pauses: decoded inside pause {bad_chunk}, retime pause split {bad_rt}; examples {examples}")
