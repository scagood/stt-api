import sys, importlib.util
import numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/wf71")
from parakeet_service import chunker as pr
spec = importlib.util.spec_from_file_location("parakeet_service.chunker_main", "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/wf71-scripts/main_chunker.py")
main = importlib.util.module_from_spec(spec); spec.loader.exec_module(main)
SR = 16000
def at(*s): return tuple(int(round(x*SR)) for x in s)
def run(mod, speech_s, total_s, target=25.0, mx=30.0, mn=20.0, ctx=5.0):
    speech = [at(a,b) for a,b in speech_s]
    mod._speech_segments = lambda _w: speech
    r = mod.plan_chunks(np.zeros(int(total_s*SR), dtype=np.float32), target_sec=target, max_sec=mx, min_sec=mn, context_sec=ctx).ranges
    return [(a/SR, b/SR) for a,b in r]
def cuts_in_speech(ranges, speech_s):
    cuts = [e for _, e in ranges[:-1]]
    return [c for c in cuts for a,b in speech_s if a < c < b]
