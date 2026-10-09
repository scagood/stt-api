"""Run F1 / new-issue repros against whichever parakeet_service is on sys.path (cwd)."""
import sys
import numpy as np

sys.path.insert(0, ".")
from parakeet_service import chunker

chunker.VAD = "volume"
chunker.VAD_GATE_DB = None
SR = chunker.TARGET_SR
BOUNDS = dict(target_sec=60.0, max_sec=75.0)


def db(x):
    return 10 ** (x / 20)


def gauss_turn(sec, level, rng):
    return rng.standard_normal(int(sec * SR)) * db(level)


def syll_turn(sec, level, floor, rng):
    # 250 ms syllables of Gaussian noise at level with 100 ms dips at floor
    parts = []
    for _ in range(round(sec / 0.35)):
        parts.append(rng.standard_normal(int(0.25 * SR)) * db(level))
        parts.append(rng.standard_normal(int(0.1 * SR)) * db(floor))
    return np.concatenate(parts)


def build(pause_sec, sounds, turn="gauss", seed=0):
    """sounds: list of (t_abs_sec, dur_sec, level_db). Turns 40 s at -20 dBFS, room tone -55."""
    rng = np.random.default_rng(seed)
    floor = rng.standard_normal(int((80 + pause_sec) * SR)) * db(-55)
    t1 = gauss_turn(40, -20, rng)
    t2 = gauss_turn(40, -20, rng)
    wav = floor.copy()
    wav[: t1.size] += t1
    off = int((40 + pause_sec) * SR)
    wav[off: off + t2.size] += t2
    for t, d, lvl in sounds:
        a = int(t * SR)
        n = int(d * SR)
        wav[a: a + n] += rng.standard_normal(n) * db(lvl)
    return wav.astype(np.float32)


def build_syll_insert(pause_sec, at, sec, level, seed=0):
    rng = np.random.default_rng(seed)
    floor = rng.standard_normal(int((80 + pause_sec) * SR)) * db(-55)
    wav = floor.copy()
    wav[: 40 * SR] += gauss_turn(40, -20, rng)
    off = int((40 + pause_sec) * SR)
    wav[off: off + 40 * SR] += gauss_turn(40, -20, rng)
    s = syll_turn(sec, level, -55, rng)
    a = int(at * SR)
    wav[a: a + s.size] += s
    return wav.astype(np.float32)


def plan(wav, ctx=0.0):
    p = chunker.plan_chunks(wav, **BOUNDS, context_sec=ctx)
    r = [(round(a / SR, 2), round(b / SR, 2)) for a, b in p.ranges]
    w = [(round(a / SR, 2), round(b / SR, 2)) for a, b in p.windows] if hasattr(p, "windows") else None
    return r, w


cases = [
    ("click 10ms -30 @44.9 in 10s", lambda: build(10, [(44.9, 0.01, -30)]), 0),
    ("knock 30ms -35 @44.9 in 10s", lambda: build(10, [(44.9, 0.03, -35)]), 0),
    ("breath 300ms -42 @42.4 in 5s", lambda: build(5, [(42.4, 0.3, -42)]), 0),
    ("breath 300ms -42 @44.9 in 10s", lambda: build(10, [(44.9, 0.3, -42)]), 0),
    ("reply 350ms -40 @44.8 in 10s", lambda: build(10, [(44.8, 0.35, -40)]), 0),
    ("reply 450ms -40 @44.8 in 10s", lambda: build(10, [(44.8, 0.45, -40)]), 0),
    ("reply 550ms -40 @44.8 in 10s", lambda: build(10, [(44.8, 0.55, -40)]), 0),
    ("reply 700ms -40 @44.8 in 10s", lambda: build(10, [(44.8, 0.7, -40)]), 0),
    ("syll turn 2s -40 @44 in 10s", lambda: build_syll_insert(10, 44.0, 2.0, -40), 0),
    ("NEW1a 15s pause, 3x250ms -42 @43,48,53", lambda: build(15, [(43, 0.25, -42), (48, 0.25, -42), (53, 0.25, -42)]), 0),
    ("NEW1b 30s pause, 7x200ms -45 every 4s @42..66 ctx5", lambda: build(30, [(42 + 4 * i, 0.2, -45) for i in range(7)]), 5),
    ("NEW1c 10s pause, 2x300ms -42 @42.6,47.0", lambda: build(10, [(42.6, 0.3, -42), (47.0, 0.3, -42)]), 0),
    ("NEW1c' 10s pause, 2x300ms -42 @43,46", lambda: build(10, [(43.0, 0.3, -42), (46.0, 0.3, -42)]), 0),
    ("NEW1c'' 10s pause, 2x300ms -42 @42.5,47.5", lambda: build(10, [(42.5, 0.3, -42), (47.5, 0.3, -42)]), 0),
]

only = sys.argv[1:] or None
for name, fn, ctx in cases:
    if only and not any(o in name for o in only):
        continue
    r, w = plan(fn(), ctx)
    print(f"{name}: ranges={r}" + (f" windows={w}" if ctx else ""))
