"""Load main's and head's chunker side by side (config is unchanged between them)."""
import importlib.util, sys
SP = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad"
sys.path.insert(0, f"{SP}/r80a-main")
import parakeet_service, parakeet_service.config  # noqa

SR = parakeet_service.config.TARGET_SR


def load(tree, tag):
    spec = importlib.util.spec_from_file_location(f"parakeet_service._r80a_{tag}", f"{SP}/{tree}/parakeet_service/chunker.py")
    m = importlib.util.module_from_spec(spec); m.__package__ = "parakeet_service"; spec.loader.exec_module(m); return m


MAIN = load("r80a-main", "main")
HEAD = load("r80a-head", "head")
for _m in (MAIN, HEAD):
    _m._speech_segments_orig = _m._speech_segments

CONFIGS = {  # target, max, context
    "v2 ctx5": (25.0, 30.0, 5.0),
    "v2 ctx0": (25.0, 30.0, 0.0),
    "v3 ctx5": (60.0, 75.0, 5.0),
}


def plan(mod, wav, cfg, segs=None, gate="keep"):
    t, mx, ctx = CONFIGS[cfg] if isinstance(cfg, str) else cfg
    if gate != "keep":
        mod.VAD_GATE_DB = gate
    mod._speech_segments = (lambda _w: segs) if segs is not None else mod._speech_segments_orig
    return mod.plan_chunks(wav, target_sec=t, max_sec=mx, min_sec=20.0, context_sec=ctx)


def load_local(path, tag):
    spec = importlib.util.spec_from_file_location(f"parakeet_service._r80a_{tag}", path)
    m = importlib.util.module_from_spec(spec); m.__package__ = "parakeet_service"; spec.loader.exec_module(m)
    m._speech_segments_orig = m._speech_segments
    return m
