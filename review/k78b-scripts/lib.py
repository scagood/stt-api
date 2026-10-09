"""Load main's and the PR's parakeet_service side by side, plus synthetic audio helpers."""
import importlib
import importlib.util
import os
import sys
import types

import numpy as np

S = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad"
ROOTS = {"main": f"{S}/k78b-main", "pr": f"{S}/k78b-head"}
SR = 16000

if "onnxruntime" not in sys.modules:
    ort = types.ModuleType("onnxruntime")
    ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    sys.modules.setdefault("onnx_asr", types.ModuleType("onnx_asr"))
    sys.modules.setdefault("onnxruntime", ort)


def load(tag):
    name = f"ps_{tag}"
    if name in sys.modules:
        return sys.modules[f"{name}.chunker"], sys.modules[f"{name}.retime"]
    root = ROOTS[tag]
    spec = importlib.util.spec_from_file_location(
        name, f"{root}/parakeet_service/__init__.py", submodule_search_locations=[f"{root}/parakeet_service"]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return importlib.import_module(f"{name}.chunker"), importlib.import_module(f"{name}.retime")


CM, RM = load("main")
CP, RP = load("pr")
MODS = {"main": (CM, RM), "pr": (CP, RP)}
BOUNDS = dict(target_sec=20.0, max_sec=40.0, context_sec=5.0)


def noise(sec, db, rng):
    return (rng.standard_normal(int(round(sec * SR))) * 10 ** (db / 20)).astype(np.float32)


def turn(sec, db, rng, syll=0.25, gap=0.08):
    """Speech-like: noise syllables with short dips, at db."""
    n = int(round(sec * SR))
    env = np.zeros(n, np.float32)
    t = 0
    while t < n:
        s = int(rng.uniform(0.6, 1.4) * syll * SR)
        env[t:t + s] = 1.0
        t += s + int(rng.uniform(0.5, 1.5) * gap * SR)
    return (rng.standard_normal(n) * 10 ** (db / 20) * env + rng.standard_normal(n) * 10 ** ((db - 40) / 20)).astype(np.float32)


def add(buf, at, sec, db, rng):
    i = int(round(at * SR)); n = int(round(sec * SR))
    n = min(n, buf.size - i)
    buf[i:i + n] += (rng.standard_normal(n) * 10 ** (db / 20)).astype(np.float32)


def plan(tag, wav, **kw):
    c = MODS[tag][0]
    b = dict(BOUNDS); b.update(kw)
    return c.plan_chunks(wav, **b)


def segs(tag, wav):
    return MODS[tag][0]._volume_speech_segments(wav)


def pauses(tag, wav):
    return MODS[tag][1].pauses(wav)


def covered(ranges, a, b):
    """Samples of [a,b) inside ranges."""
    return sum(max(0, min(e, b) - max(s, a)) for s, e in ranges)


def sec(r):
    return [(round(a / SR, 2), round(b / SR, 2)) for a, b in r]


def loud(tag, wav, ratio=0.4, relisten_sec=3.0):
    c = MODS[tag][0]
    rms = c.frame_rms(wav)
    return c.loud_frames(rms, ratio, max(1, int(relisten_sec * SR) // c.FRAME))


def loud_runs_sec(tag, wav, lo, hi, **kw):
    lf = loud(tag, wav, **kw)
    r = MODS[tag][0].runs(lf)
    return [(round(a * 0.02, 2), round(b * 0.02, 2)) for a, b in r.tolist() if b * 0.02 > lo and a * 0.02 < hi]
