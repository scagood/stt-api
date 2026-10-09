import sys, importlib
import numpy as np
SR = 16000
from parakeet_service import chunker as new
old = importlib.import_module("parakeet_service._chunker_old")
main = importlib.import_module("parakeet_service._chunker_main")
VERS = {"new": new, "old": old, "main": main}
BOUNDS = {
    "v2": dict(target_sec=25.0, max_sec=30.0, min_sec=20.0, context_sec=5.0),
    "v3": dict(target_sec=60.0, max_sec=75.0, min_sec=20.0, context_sec=5.0),
    "wh": dict(target_sec=25.0, max_sec=30.0, min_sec=20.0, context_sec=0.0),
}
def at(*s): return tuple(int(round(x * SR)) for x in s)
class Fake:
    def __init__(self, n): self.size = n
def plan(mod, model, speech_s=None, seconds=None, speech=None, total=None):
    if speech is None:
        speech = [at(*p) for p in speech_s]
    if total is None:
        total = at(seconds)[0]
    orig = mod._speech_segments
    mod._speech_segments = lambda _w: list(speech)
    try:
        wav = np.broadcast_to(np.float32(0), (total,))
        p = mod.plan_chunks(wav, **BOUNDS[model])
    finally:
        mod._speech_segments = orig
    return p
def fmt(rs): return [(round(a / SR, 3), round(b / SR, 3)) for a, b in rs]
def show(model, speech_s, seconds):
    for k, m in VERS.items():
        print(f"  {k:5s} {fmt(plan(m, model, speech_s, seconds).ranges)}")
fix1 = importlib.import_module("parakeet_service._chunker_fix1")
fix2 = importlib.import_module("parakeet_service._chunker_fix2")
fix3 = importlib.import_module("parakeet_service._chunker_fix3")
