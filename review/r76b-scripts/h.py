import os, sys, importlib.util, random
S = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad"
sys.path.insert(0, S + "/r76b-pr")
import parakeet_service, parakeet_service.config
SR = parakeet_service.config.TARGET_SR
def load(name):
    spec = importlib.util.spec_from_file_location(f"parakeet_service._r76b_{name}", f"{S}/r76b-scripts/chunker_{name}.py")
    mod = importlib.util.module_from_spec(spec); mod.__package__ = "parakeet_service"
    spec.loader.exec_module(mod); return mod
MODS = {n: load(n) for n in os.environ.get("NAMES", "main,pr").split(",")}
CONFIGS = {
  "v2c5": dict(target=25.0, mx=30.0, ctx=5.0, mn=20.0),
  "v2c0": dict(target=25.0, mx=30.0, ctx=0.0, mn=20.0),
  "v3c5": dict(target=60.0, mx=75.0, ctx=5.0, mn=20.0),
  "whis": dict(target=25.0, mx=30.0, ctx=0.0, mn=20.0),
}
def own_of(cfg): return cfg["mx"] - 2 * cfg["ctx"]
class W:
    def __init__(self, n): self.size = n
def run_s(mod, segs, total, cfg):
    """segs, total in samples."""
    mod._speech_segments = lambda _w: segs
    plan = mod.plan_chunks(W(total), target_sec=cfg["target"], max_sec=cfg["mx"], min_sec=cfg["mn"], context_sec=cfg["ctx"])
    return plan.ranges, plan.windows, plan.speech
def to_s(segs_sec, total_sec):
    return [(int(round(a * SR)), int(round(b * SR))) for a, b in segs_sec], int(round(total_sec * SR))
def run(mod, segs_sec, total_sec, cfg):
    segs, total = to_s(segs_sec, total_sec)
    r, w, sp = run_s(mod, segs, total, cfg)
    return r, w, sp, total
def inside(x, segs):
    return any(s < x < e for s, e in segs)
def metrics(ranges, windows, speech, total, cfg):
    own = int(own_of(cfg) * SR); mx = int(cfg["mx"] * SR)
    segs = speech
    bad = []
    if any(not (0 <= a < b <= total) for a, b in ranges): bad.append("outside")
    if any(b > c for (a, b), (c, d) in zip(ranges, ranges[1:])): bad.append("overlap")
    if total > mx and any(b - a > own for a, b in ranges): bad.append("range>own")
    if any(b - a > mx for a, b in windows): bad.append("window>max")
    if any(not (wa <= a and b <= wb) for (a, b), (wa, wb) in zip(ranges, windows)): bad.append("window!>=range")
    isc = sum(1 for (a, b), (c, d) in zip(ranges, ranges[1:]) if b == c and inside(b, segs))
    silent = sum(1 for a, b in ranges if not any(s < b and e > a for s, e in segs))
    short = sum(1 for a, b in ranges if b - a < 2 * SR)
    lost = 0
    for s, e in segs:
        cov = sum(max(0, min(b, e) - max(a, s)) for a, b in ranges)
        lost += max(0, (e - s) - cov)
    wedge = 0
    for (a, b), (wa, wb) in zip(ranges, windows):
        if wa != a and inside(wa, segs): wedge += 1
        if wb != b and inside(wb, segs): wedge += 1
    # silence decoded inside ranges (samples)
    sil = sum(b - a for a, b in ranges) - sum(max(0, min(b, e) - max(a, s)) for a, b in ranges for s, e in segs)
    return dict(n=len(ranges), isc=isc, silent=silent, short=short, lost=int(lost > 0), bad=int(bool(bad)), wedge=wedge, sil=sil, badl=bad)
def sec(r): return [(round(a / SR, 3), round(b / SR, 3)) for a, b in r]
def lens(r): return [round((b - a) / SR, 2) for a, b in r]
if os.environ.get("TRIM"):
    for _m in MODS.values(): _m.CHUNK_TRIM_SILENCE_SEC = float(os.environ["TRIM"])
CONFIGS.update({
  "v2c25": dict(target=25.0, mx=30.0, ctx=2.5, mn=20.0),
  "v2c75": dict(target=25.0, mx=30.0, ctx=7.5, mn=20.0),
  "v2c5m10": dict(target=25.0, mx=30.0, ctx=5.0, mn=10.0),
  "v2c0m10": dict(target=25.0, mx=30.0, ctx=0.0, mn=10.0),
  "v3c15": dict(target=60.0, mx=75.0, ctx=15.0, mn=20.0),
  "v3c0": dict(target=60.0, mx=75.0, ctx=0.0, mn=20.0),
})
