import sys
sys.path.insert(0, ".")
import numpy as np
from parakeet_service import chunker
SR = chunker.TARGET_SR
for name, pause_sec, sounds in [("click", 10, [(4.9, 0.01, -30)]), ("knock", 10, [(4.9, 0.03, -35)]), ("breath", 5, [(2.4, 0.3, -42)])]:
    rng = np.random.default_rng(1)
    pause = rng.standard_normal(int(pause_sec * SR)) * 10 ** (-55 / 20)
    base = chunker.frame_rms(pause.astype(np.float32))
    for at, seconds, level_db in sounds:
        n = int(seconds * SR)
        pause[int(at * SR): int(at * SR) + n] += rng.standard_normal(n) * 10 ** (level_db / 20)
    rms = chunker.frame_rms(pause.astype(np.float32))
    f0 = int(sounds[0][0] * SR) // chunker.FRAME
    nf = -(-int(sounds[0][1] * SR) // chunker.FRAME)
    med = np.median(base)
    print(name, "frame offset", int(sounds[0][0] * SR) % chunker.FRAME, "dB over median room tone:",
          [round(20 * np.log10(rms[f0 + i] / med), 1) for i in range(min(nf, 4))], "(nominal level - tone:", sounds[0][2] + 55, "dB)")
