import sys, importlib.util
import numpy as np
S="/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad"
def load(name, path):
    # load chunker from a given tree as its own package
    import importlib
    spec = importlib.util.spec_from_file_location(name, path + "/parakeet_service/__init__.py", submodule_search_locations=[path + "/parakeet_service"])
    pkg = importlib.util.module_from_spec(spec); sys.modules[name] = pkg; spec.loader.exec_module(pkg)
    spec2 = importlib.util.spec_from_file_location(name + ".chunker", path + "/parakeet_service/chunker.py")
    m = importlib.util.module_from_spec(spec2); sys.modules[name + ".chunker"] = m; spec2.loader.exec_module(m)
    return m
pr = load("pr_pkg", S + "/wf71")
fx = load("fx_pkg", S + "/wf71-scripts/fixF1")
spec = importlib.util.spec_from_file_location("pr_pkg.chunker_main", S + "/wf71-scripts/main_chunker.py")
mn = importlib.util.module_from_spec(spec); spec.loader.exec_module(mn)
SR = 16000
OWN = 20 * SR
def stats(mod, speech, total, **kw):
    mod._speech_segments = lambda _w: speech
    r = mod.plan_chunks(np.broadcast_to(np.float32(0), (total,)), target_sec=25.0, max_sec=30.0, min_sec=20.0, context_sec=5.0).ranges
    cuts = [e for _, e in r[:-1] if any(nr[0] == e for nr in r)]
    inside = [c for c in cuts for a, b in speech if a < c < b]
    avoidable = [c for c in cuts for a, b in speech if a < c < b and b - a <= OWN]
    sliver = [x for x in r if not any(a < x[1] and x[0] < b for a, b in speech)]
    return len(r), len(inside), len(avoidable), len(sliver)

rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
tot = {k: np.zeros(4, int) for k in ("main", "pr", "fix")}
for trial in range(3000):
    seconds = rng.uniform(40, 200)
    speech, at = [], rng.uniform(0, 4)
    while True:
        length = rng.choice([rng.uniform(1, 8), rng.uniform(8, 22)], p=[0.7, 0.3])
        if at + length > seconds - 0.5: break
        speech.append((int(at * SR), int((at + length) * SR)))
        at += length + rng.choice([rng.uniform(0.2, 2.0), rng.uniform(2.0, 5.0)], p=[0.85, 0.15])
    if not speech: continue
    total = int(seconds * SR)
    for k, m in (("main", mn), ("pr", pr), ("fix", fx)):
        tot[k] += stats(m, speech, total)
for k, v in tot.items():
    print(k, "chunks=%d inside=%d avoidable=%d slivers=%d" % tuple(v))
