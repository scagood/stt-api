import importlib.util, sys, types
import numpy as np
S = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad"

def load(tag):
    for k in [k for k in sys.modules if k.startswith("parakeet_service")]:
        del sys.modules[k]
    sys.path.insert(0, f"{S}/m75a-{tag}")
    import parakeet_service.chunker as c, parakeet_service.retime as r
    sys.path.pop(0)
    c.VAD = "volume"; c.VAD_GATE_DB = -40.0
    return c, r

old_c, old_r = load("old")
new_c, new_r = load("new")
assert old_c is not new_c
rng = np.random.default_rng(0)
SR = 16000
bad = 0
for trial in range(150):
    # piecewise-level random audio: segments of random level and duration
    parts = []
    for _ in range(rng.integers(3, 25)):
        dur = float(rng.choice([0.01, 0.03, 0.1, 0.3, 0.6, 1.5, 4, 8, 20]))
        lvl = float(rng.uniform(-75, -10))
        parts.append(rng.standard_normal(int(dur * SR)) * 10 ** (lvl / 20))
    wav = np.concatenate(parts).astype(np.float32)
    rms = old_c.frame_rms(wav)
    for ratio, rel in [(0.4, 150), (0.6, 150), (0.4, 25), (0.6, 5)]:
        a = old_c.loud_frames(rms, ratio, rel); b = new_c.loud_frames(rms, ratio, rel)
        if not np.array_equal(a, b):
            bad += 1; print("loud_frames differ", trial, ratio, rel)
    if old_c._volume_speech_segments(wav) != new_c._volume_speech_segments(wav):
        bad += 1; print("vad differs", trial)
    if old_r.pauses(wav) != new_r.pauses(wav):
        bad += 1; print("pauses differ", trial)
print("trials 300, mismatches", bad)
