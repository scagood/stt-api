import os, sys, importlib.util
S = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad"
sys.path.insert(0, f"{S}/r76a-head")
import numpy as np
import parakeet_service, parakeet_service.config
SR = parakeet_service.config.TARGET_SR
SRC = {"main": f"{S}/r76a-main/parakeet_service/chunker.py", "head": os.environ.get("HEADSRC", f"{S}/r76a-head/parakeet_service/chunker.py")}
def load(name, path=None):
    spec = importlib.util.spec_from_file_location(f"parakeet_service._r76a_{name}", path or SRC[name])
    mod = importlib.util.module_from_spec(spec); mod.__package__ = "parakeet_service"
    spec.loader.exec_module(mod); return mod
def run(mod, segs_sec, total_sec, target=25.0, mx=30.0, ctx=5.0, mn=20.0, samples=False):
    total = int(round(total_sec * SR)) if not samples else total_sec
    segs = [(int(round(a * SR)), int(round(b * SR))) for a, b in segs_sec] if not samples else list(segs_sec)
    mod._speech_segments = lambda _w: segs
    wav = np.broadcast_to(np.float32(0), (total,))
    plan = mod.plan_chunks(wav, target_sec=target, max_sec=mx, min_sec=mn, context_sec=ctx)
    return plan.ranges, plan.windows, segs, total
def in_speech_cuts(ranges, segs):
    return sum(1 for (a, b), (c, d) in zip(ranges, ranges[1:]) if b == c and any(s < b < e for s, e in segs))
def silent_ranges(ranges, segs):
    return sum(1 for a, b in ranges if not any(s < b and e > a for s, e in segs))
def speech_lost(ranges, segs):
    lost = 0
    for s, e in segs:
        cov = sum(max(0, min(b, e) - max(a, s)) for a, b in ranges)
        lost += (e - s) - cov
    return lost
def sec(ranges):
    return [(round(a / SR, 3), round(b / SR, 3)) for a, b in ranges]
