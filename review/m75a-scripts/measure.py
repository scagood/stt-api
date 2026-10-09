import sys
sys.path.insert(0, ".")
import numpy as np
from parakeet_service import chunker
SR = chunker.TARGET_SR
for pause_sec, seconds, level_db in [(10, 0.01, -30), (10, 0.03, -35), (5, 0.3, -42)]:
    rng = np.random.default_rng(1)
    pause = rng.standard_normal(pause_sec * SR) * 10 ** (-55 / 20)
    base = chunker.frame_rms(pause.astype(np.float32))
    at = int((pause_sec / 2 - 0.1) * SR)
    pause[at: at + int(seconds * SR)] += rng.standard_normal(int(seconds * SR)) * 10 ** (level_db / 20)
    rms = chunker.frame_rms(pause.astype(np.float32))
    f0 = at // chunker.FRAME
    nf = -(-int(seconds * SR) // chunker.FRAME)
    tone_med = np.median(base)
    tone_p10 = np.percentile(rms, 10)
    over = [round(20 * np.log10(rms[f0 + i] / tone_med), 1) for i in range(nf)]
    overp10 = [round(20 * np.log10(rms[f0 + i] / tone_p10), 1) for i in range(nf)]
    print(pause_sec, seconds, level_db, "at frame offset", at % chunker.FRAME, "dB over median room tone:", over[:3], "...", "over p10:", overp10[:3])
