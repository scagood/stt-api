import sys, importlib
import numpy as np
S = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad"
TREES = {"main": S + "/v78-main", "old": S + "/v78-old", "head": S + "/v78-head"}

def load(name):
    for k in list(sys.modules):
        if k == "parakeet_service" or k.startswith("parakeet_service."):
            del sys.modules[k]
    sys.path.insert(0, TREES[name])
    try:
        ch = importlib.import_module("parakeet_service.chunker")
        rt = importlib.import_module("parakeet_service.retime")
    finally:
        sys.path.pop(0)
    ch.VAD = "volume"; ch.VAD_GATE_DB = None
    return ch, rt

MODS = {n: load(n) for n in TREES}
for n,(c,r) in MODS.items():
    assert TREES[n] in c.__file__, (n, c.__file__)
SR = 16000
BOUNDS = dict(target_sec=60.0, max_sec=75.0)

def db(x): return 10 ** (x / 20)

def build(pause_sec, sounds, seed=0, turn_sec=40, turn_db=-20, floor_db=-55):
    rng = np.random.default_rng(seed)
    total = 2 * turn_sec + pause_sec
    wav = rng.standard_normal(int(total * SR)) * db(floor_db)
    wav[: int(turn_sec * SR)] += rng.standard_normal(int(turn_sec * SR)) * db(turn_db)
    off = int((turn_sec + pause_sec) * SR)
    wav[off: off + int(turn_sec * SR)] += rng.standard_normal(int(turn_sec * SR)) * db(turn_db)
    for t, d, lvl in sounds:
        a = int(t * SR); n = int(d * SR)
        wav[a: a + n] += rng.standard_normal(n)[: wav[a:a+n].size] * db(lvl)
    return wav.astype(np.float32)

def secs(r): return [(round(a / SR, 2), round(b / SR, 2)) for a, b in r]

def plan(name, wav, ctx=0.0, **kw):
    ch, _ = MODS[name]
    b = dict(BOUNDS); b.update(kw)
    return ch.plan_chunks(wav, **b, context_sec=ctx)

def pauses(name, wav):
    return MODS[name][1].pauses(wav)
