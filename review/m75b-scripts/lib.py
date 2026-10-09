import sys, importlib
import numpy as np
S = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad"
TREES = {"main": S + "/m75b-main", "prev": S + "/m75b-prev", "head": S + "/m75b-head"}

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
SR = 16000
BOUNDS = dict(target_sec=60.0, max_sec=75.0)

def db(x): return 10 ** (x / 20)

def build(pause_sec, sounds, seed=0, turn_sec=40, turn_db=-20, floor_db=-55, first=True, last=True):
    """sounds: (abs t sec, dur sec, dBFS) Gaussian bursts added. Turns at [0,turn) and [turn+pause, 2turn+pause)."""
    rng = np.random.default_rng(seed)
    total = (turn_sec if first else 0) + pause_sec + (turn_sec if last else 0)
    wav = rng.standard_normal(int(total * SR)) * db(floor_db)
    if first:
        wav[: int(turn_sec * SR)] += rng.standard_normal(int(turn_sec * SR)) * db(turn_db)
    if last:
        off = int(((turn_sec if first else 0) + pause_sec) * SR)
        wav[off: off + int(turn_sec * SR)] += rng.standard_normal(int(turn_sec * SR)) * db(turn_db)
    for t, d, lvl in sounds:
        a = int(t * SR); n = int(d * SR)
        wav[a: a + n] += rng.standard_normal(n)[: wav[a:a+n].size] * db(lvl)
    return wav.astype(np.float32)

def rng_(r): return [(round(a / SR, 2), round(b / SR, 2)) for a, b in r]

def plan(name, wav, ctx=5.0):
    ch, _ = MODS[name]
    p = ch.plan_chunks(wav, **BOUNDS, context_sec=ctx)
    return rng_(p.ranges)

def pauses(name, wav):
    _, rt = MODS[name]
    return [(round(a, 2), round(b, 2)) for a, b in rt.pauses(wav)]

def loudmask(name, rms, ratio, relisten):
    ch, _ = MODS[name]
    if hasattr(ch, "loud_frames"):
        return ch.loud_frames(rms, ratio, relisten)
    return rms > ch.relative_gate(rms, ratio)

def report(label, wav, ctx=5.0, show_pauses=False, window=None):
    print("==", label)
    for n in ("main", "prev", "head"):
        print(f"  {n:5s} plan:", plan(n, wav, ctx))
    if show_pauses:
        for n in ("main", "prev", "head"):
            p = pauses(n, wav)
            if window:
                p = [x for x in p if x[1] > window[0] and x[0] < window[1]]
            print(f"  {n:5s} pauses:", p)
