import importlib.util, sys, types
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/wf74")
import numpy as np
import parakeet_service  # from worktree cwd
import parakeet_service.config
S = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/wf74-scripts"
SR = parakeet_service.config.TARGET_SR

def load(name):
    spec = importlib.util.spec_from_file_location(f"parakeet_service._cmp_{name}", f"{S}/chunker_{name}.py")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "parakeet_service"
    spec.loader.exec_module(mod)
    return mod

MODS = {n: load(n) for n in ("55", "74", "fix")}

def run(mod, segs_sec, total_sec, target=25.0, mx=30.0, ctx=5.0, mn=20.0):
    total = int(round(total_sec * SR))
    segs = [(int(round(a * SR)), int(round(b * SR))) for a, b in segs_sec]
    mod._speech_segments = lambda _w: segs
    wav = np.zeros(total, dtype=np.float32)
    plan = mod.plan_chunks(wav, target_sec=target, max_sec=mx, min_sec=mn, context_sec=ctx)
    return plan, segs, total

def in_speech_cuts(ranges, segs):
    n = 0
    for (a, b), (c, d) in zip(ranges, ranges[1:]):
        if b == c and any(s < b < e for s, e in segs):
            n += 1
    return n

def show(ranges):
    return [round((b - a) / SR, 2) for a, b in ranges]
