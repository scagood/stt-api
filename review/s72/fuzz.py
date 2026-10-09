import sys, importlib.util, types, numpy as np
sys.path.insert(0, sys.argv[1])
import parakeet_service.chunker as new
import parakeet_service.retime as newr
# load old chunker as a module inside the package namespace
src = open(sys.argv[2]).read()
old = types.ModuleType("parakeet_service.old_chunker"); old.__package__ = "parakeet_service"
exec(compile(src, "old_chunker", "exec"), old.__dict__)
srcr = open(sys.argv[3]).read().replace("from .chunker import", "from .old_chunker import")
sys.modules["parakeet_service.old_chunker"] = old
oldr = types.ModuleType("parakeet_service.old_retime"); oldr.__package__ = "parakeet_service"
exec(compile(srcr, "old_retime", "exec"), oldr.__dict__)
rng = np.random.default_rng(1)
bad = 0
for trial in range(20000):
    n = int(rng.integers(0, 400)) * new.FRAME + int(rng.integers(0, new.FRAME))
    frames = (rng.random(n // new.FRAME + 1) < rng.random()).repeat(new.FRAME)[:n]
    wav = (rng.normal(size=n) * np.where(frames, 0.3, 0.001)).astype(np.float32)
    gate = None if trial % 2 else float(rng.uniform(-70, -10))
    ms = int(rng.choice([0, 10, 20, 30, 100, 400, 1000]))
    pad = int(rng.choice([0, 30, 200]))
    for m in (new, old):
        m.VAD_GATE_DB = gate; m.VAD_MIN_SILENCE_MS = ms; m.VAD_SPEECH_PAD_MS = pad
    a, b = new._volume_speech_segments(wav), old._volume_speech_segments(wav)
    if a != b:
        bad += 1
        if bad < 5: print("VAD diff", n, gate, ms, pad, a[:3], b[:3])
    if newr.pauses(wav) != oldr.pauses(wav):
        bad += 1
        if bad < 5: print("pauses diff", n)
print("mismatches", bad)
