import importlib.util, sys
import numpy as np
S = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad"
sys.path.insert(0, S + "/rr71")
import parakeet_service.chunker as new
def load(name, path):
    spec = importlib.util.spec_from_file_location("parakeet_service." + name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m); return m
old = load("chunker_old", S + "/rr71-scripts/chunker_55df37c.py")
main = load("chunker_main", S + "/rr71-scripts/chunker_origin_main.py")
VERS = {"new": new, "old": old, "main": main}
SR = new.TARGET_SR
BOUNDS = {"v2": dict(target_sec=25.0, max_sec=30.0, min_sec=20.0, context_sec=5.0),
          "v3": dict(target_sec=60.0, max_sec=75.0, min_sec=20.0, context_sec=5.0),
          "wh": dict(target_sec=25.0, max_sec=30.0, min_sec=20.0, context_sec=0.0)}
def plan(mod, speech_samples, total, b):
    mod._speech_segments = lambda _w: speech_samples
    return mod.plan_chunks(np.broadcast_to(np.float32(0), (total,)), **b)
def run(mod, speech_sec, seconds, b="v2"):
    sp = [tuple(int(x * SR) for x in s) for s in speech_sec]
    p = plan(mod, sp, int(seconds * SR), BOUNDS[b])
    return [(a / SR, e / SR) for a, e in p.ranges]
