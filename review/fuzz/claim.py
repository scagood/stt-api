import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from parakeet_service import chunker
SR = 16000
def _at(*s): return tuple(int(x*SR) for x in s)
def ordinary(seconds):
    rng = np.random.default_rng(0); speech, at = [], 1.0
    while True:
        length = rng.uniform(2, 6)
        if at + length > seconds - 1: return speech, int(seconds*SR)
        speech.append(_at(at, at+length)); at += length + rng.uniform(0.5, 1.5)
for secs in (600, 7200):
    speech, total = ordinary(secs)
    chunker._speech_segments = lambda _w: speech
    r = chunker.auto_chunk(np.broadcast_to(np.float32(0), (total,)), target_sec=25.0, max_sec=30.0, min_sec=20.0, context_sec=5.0)
    print(secs, len(r), sum(1 for s, e in r[:-1] for a, b in speech if a < e < b), min(e-s for s, e in r)/SR)
