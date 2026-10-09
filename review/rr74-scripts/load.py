import os, sys; sys.path.insert(0, os.getcwd())
import importlib.util, sys
import numpy as np
import parakeet_service, parakeet_service.config
S = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/rr74-scripts"
SR = parakeet_service.config.TARGET_SR
def load(name):
    spec = importlib.util.spec_from_file_location(f"parakeet_service._cmp_{name}", f"{S}/chunker_{name}.py")
    mod = importlib.util.module_from_spec(spec); mod.__package__ = "parakeet_service"
    spec.loader.exec_module(mod); return mod
NAMES = tuple(os.environ.get("NAMES", "origin_main,55df37c,74d1902,c032ee4").split(","))
MODS = {n: load(n) for n in NAMES}
def run(mod, segs_sec, total_sec, target=25.0, mx=30.0, ctx=5.0, mn=20.0):
    total = int(round(total_sec * SR))
    segs = [(int(round(a * SR)), int(round(b * SR))) for a, b in segs_sec]
    mod._speech_segments = lambda _w: segs
    wav = np.zeros(total, dtype=np.float32)
    if hasattr(mod, "plan_chunks"):
        plan = mod.plan_chunks(wav, target_sec=target, max_sec=mx, min_sec=mn, context_sec=ctx)
        return plan.ranges, plan.windows, segs, total
    r = mod.auto_chunk(wav, target_sec=target, max_sec=mx, min_sec=mn)
    return r, r, segs, total
def in_speech_cuts(ranges, segs):
    return sum(1 for (a, b), (c, d) in zip(ranges, ranges[1:]) if b == c and any(s < b < e for s, e in segs))
def silent_ranges(ranges, segs):
    return sum(1 for a, b in ranges if not any(s < b and e > a for s, e in segs))
def speech_lost(ranges, segs):
    lost = 0
    for s, e in segs:
        cov = 0
        for a, b in ranges:
            cov += max(0, min(b, e) - max(a, s))
        lost += (e - s) - cov
    return lost
def show(ranges):
    return [round((b - a) / SR, 2) for a, b in ranges]
def sec(ranges):
    return [(round(a / SR, 2), round(b / SR, 2)) for a, b in ranges]
