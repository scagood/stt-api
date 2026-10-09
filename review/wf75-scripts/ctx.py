import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from parakeet_service import chunker
chunker.VAD = "volume"; chunker.VAD_GATE_DB = None
SR = chunker.TARGET_SR
def db(x): return 10 ** (x / 20)
rng = np.random.default_rng(0)
wav = rng.standard_normal(90 * SR) * db(-55)
wav[: 40 * SR] = rng.standard_normal(40 * SR) * db(-20)
wav[50 * SR:] = rng.standard_normal(40 * SR) * db(-20)
a = int(44.9 * SR); wav[a:a + 160] = rng.standard_normal(160) * db(float(sys.argv[2]))
p = chunker.plan_chunks(wav.astype(np.float32), target_sec=60.0, max_sec=75, context_sec=5)
f = lambda rs: [(round(x / SR, 2), round(y / SR, 2)) for x, y in rs]
print(f(p.ranges)); print(f(p.windows))
