import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from parakeet_service import chunker
chunker.VAD = "volume"; chunker.VAD_GATE_DB = None
SR = chunker.TARGET_SR
def db(x): return 10 ** (x / 20)
def build(pause_sec, noise_sec, noise_db, offset=0, seed=0):
    rng = np.random.default_rng(seed)
    wav = rng.standard_normal(int((80 + pause_sec) * SR)) * db(-55)
    wav[: 40 * SR] = rng.standard_normal(40 * SR) * db(-20)
    wav[(40 + pause_sec) * SR:] = rng.standard_normal(40 * SR) * db(-20)
    a = int((40 + pause_sec / 2) * SR) + offset; n = int(noise_sec * SR)
    wav[a: a + n] = rng.standard_normal(n) * db(noise_db)
    return wav.astype(np.float32)
for nsec in (0.01, 0.02):
    for lvl in np.arange(-36, -28, 0.5):
        r = chunker.plan_chunks(build(10, nsec, lvl), target_sec=60.0, max_sec=75, context_sec=0).ranges
        print(nsec, lvl, len(r))
# frame rms of room tone
rms = chunker.frame_rms(build(10, 0, -30))
part = rms[int(41*SR/320):int(49*SR/320)]
print("room tone mean dB", 20*np.log10(part.mean()), "p10 dB", 20*np.log10(np.percentile(part,10)))
