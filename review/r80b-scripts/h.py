"""Harness: load main and PR chunkers, synthesise speech-like audio."""
import os, sys, importlib.util
import numpy as np
D = os.path.dirname(os.path.abspath(__file__))
ROOT = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/r80b-head"
sys.path.insert(0, ROOT)
import parakeet_service, parakeet_service.config
SR = parakeet_service.config.TARGET_SR


def load(name):
    spec = importlib.util.spec_from_file_location(f"parakeet_service._r80bh_{name}", f"{D}/chunker_{name}.py")
    m = importlib.util.module_from_spec(spec); m.__package__ = "parakeet_service"; spec.loader.exec_module(m)
    return m


NAMES = tuple(os.environ.get("NAMES", "main,pr").split(","))
M = {n: load(n) for n in NAMES}

CONFIGS = {
    "v2 ctx5": dict(target=25.0, mx=30.0, ctx=5.0),
    "v3 ctx5": dict(target=60.0, mx=75.0, ctx=5.0),
    "whisper": dict(target=25.0, mx=30.0, ctx=0.0),
}


def db(x):
    return 10.0 ** (x / 20.0)


def word_signal(rng, n, level):
    """A harmonic + noise burst of n samples at RMS ~level with 15 ms ramps."""
    t = np.arange(n) / SR
    f0 = rng.uniform(90, 220)
    sig = np.zeros(n)
    for h in range(1, 12):
        sig += np.sin(2 * np.pi * f0 * h * t + rng.uniform(0, 6.28)) / h
    sig += 0.5 * rng.standard_normal(n)
    # syllabic modulation
    syl = rng.uniform(3, 7)
    env = 0.6 + 0.4 * np.sin(2 * np.pi * syl * t + rng.uniform(0, 6.28))
    r = min(int(0.015 * SR), n // 2)
    if r > 0:
        env[:r] *= np.linspace(0, 1, r); env[n - r:] *= np.linspace(1, 0, r)
    sig *= env
    rms = np.sqrt(np.mean(sig ** 2)) + 1e-12
    return sig / rms * level


def synth(rng, segs, total, speech_db=-20.0, gap_db=(-30, -10), room_db=-60.0,
          word=(0.15, 0.5), gap=(0.05, 0.35), closures=None):
    """Audio of `total` samples: room tone, and inside each (sample) segment
    words with gaps. Returns wav (float32) and word list [(a, b)] in samples.
    closures: None or dict(p=prob per word, dur=(lo,hi), depth_db=x) adds an
    internal dip inside words (a stop closure) -- still part of the word."""
    wav = rng.standard_normal(total) * db(room_db)
    words = []
    lvl = db(speech_db)
    for a, b in segs:
        t = a
        while t < b:
            wl = int(rng.uniform(*word) * SR)
            e = min(b, t + wl)
            if e - t < int(0.05 * SR):
                break
            s = word_signal(rng, e - t, lvl)
            if closures and rng.random() < closures["p"] and e - t > int(0.25 * SR):
                cd = int(rng.uniform(*closures["dur"]) * SR)
                c0 = int(rng.uniform(0.3, 0.7) * (e - t - cd))
                s[c0:c0 + cd] *= db(closures["depth_db"])
            wav[t:e] += s
            words.append((t, e))
            g = int(rng.uniform(*gap) * SR)
            gd = rng.uniform(*gap_db) if isinstance(gap_db, tuple) else gap_db
            # gap: low-level noise (coarticulation) gd dB under speech
            ge = min(b, e + g)
            if ge > e:
                wav[e:ge] += rng.standard_normal(ge - e) * lvl * db(gd)
            t = ge
    return wav.astype(np.float32), words


def plan(mod, wav, segs, cfg, mn=20.0):
    mod._speech_segments = lambda _w: list(segs)
    p = mod.plan_chunks(wav, target_sec=cfg["target"], max_sec=cfg["mx"], min_sec=mn, context_sec=cfg["ctx"])
    return p.ranges, p.windows


def cuts_in_words(ranges, segs, words):
    """Forced cuts (meeting ranges, cut inside a VAD segment): (count, inside a word)."""
    starts = np.array([w[0] for w in words]); ends = np.array([w[1] for w in words])
    n = inw = 0
    for (a, b), (c, d) in zip(ranges, ranges[1:]):
        if b != c or not any(s < b < e for s, e in segs):
            continue
        n += 1
        i = np.searchsorted(starts, b, side="right") - 1
        if i >= 0 and starts[i] < b < ends[i]:
            inw += 1
    return n, inw


def covered(spans, total):
    m = np.zeros(total, bool)
    for a, b in spans:
        m[a:b] = True
    return m


def check_bounds(ranges, windows, total, cfg):
    own = int((cfg["mx"] - 2 * cfg["ctx"]) * SR); mx = int(cfg["mx"] * SR)
    bad = []
    if any(not (0 <= a < b <= total) for a, b in ranges): bad.append("outside")
    if any(b > c for (a, b), (c, d) in zip(ranges, ranges[1:])): bad.append("overlap")
    if any(b - a > own for a, b in ranges): bad.append("range>own")
    if any(b - a > mx for a, b in windows): bad.append("window>max")
    if any(not (wa <= a and b <= wb) for (a, b), (wa, wb) in zip(ranges, windows)): bad.append("window!>=range")
    return bad


def sec(r):
    return [(round(a / SR, 2), round(b / SR, 2)) for a, b in r]


def cuts_in_words_all(ranges, words):
    """Every meeting boundary inside a word (whether or not in a VAD segment)."""
    starts = np.array([w[0] for w in words]); ends = np.array([w[1] for w in words])
    n = 0
    for (a, b), (c, d) in zip(ranges, ranges[1:]):
        if b != c:
            continue
        i = np.searchsorted(starts, b, side="right") - 1
        n += bool(i >= 0 and starts[i] < b < ends[i])
    return n
