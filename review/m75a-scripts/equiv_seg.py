"""_volume_speech_segments refactor equivalence: e8811de (old) vs 308d68b (new)."""
import sys
import numpy as np
S = sys.argv[1]

def load(tag):
    for k in [k for k in sys.modules if k.startswith("parakeet_service")]:
        del sys.modules[k]
    sys.path.insert(0, f"{S}/m75a-{tag}")
    import parakeet_service.chunker as c
    sys.path.pop(0)
    c.VAD = "volume"; c.VAD_GATE_DB = None
    return c

old = load("old"); new = load("new")
assert old is not new and old.__file__ != new.__file__
print("old", old.__file__, "\nnew", new.__file__)
orig_new_loud = new.loud_frames
SR = 16000

def rand_wav(rng):
    parts = []
    for _ in range(rng.integers(1, 25)):
        dur = float(rng.choice([0.001, 0.01, 0.03, 0.1, 0.3, 0.42, 0.6, 1.5, 4, 8, 20]))
        lvl = float(rng.uniform(-90, -10))
        parts.append(rng.standard_normal(max(1, int(dur * SR))) * 10 ** (lvl / 20))
    return np.concatenate(parts).astype(np.float32)

special = [np.zeros(0, np.float32), np.zeros(10, np.float32), np.zeros(319, np.float32), np.zeros(320, np.float32),
           np.zeros(16000 * 30, np.float32), np.full(16000 * 5, 0.5, np.float32), (np.ones(100) * 0.01).astype(np.float32),
           (np.ones(100) * 1e-5).astype(np.float32), np.random.default_rng(9).standard_normal(16000 * 10).astype(np.float32) * 1e-6]

def compare(label, gate, min_sil, patch_loud):
    rng = np.random.default_rng(0)
    for c in (old, new):
        c.VAD_GATE_DB = gate; c.VAD_MIN_SILENCE_MS = min_sil
    new.loud_frames = old.loud_frames if patch_loud else orig_new_loud
    bad = 0; n = 0; nonempty = 0
    for wav in special + [rand_wav(rng) for _ in range(300)]:
        a = old._volume_speech_segments(wav); b = new._volume_speech_segments(wav)
        n += 1; nonempty += bool(a)
        if a != b:
            bad += 1
    print(f"{label} gate={gate} min_silence_ms={min_sil} loud_frames={'old' if patch_loud else 'new'}: {n} wavs ({nonempty} with speech), mismatches {bad}")

for ms in (400, 20, 1, 1000):
    compare("fixed", -40.0, ms, False)
    compare("relative, old loud_frames in both", None, ms, True)
compare("relative, own loud_frames (expected to differ)", None, 400, False)
new.loud_frames = orig_new_loud

# _joined vs the old inline code on random run arrays
rng = np.random.default_rng(1); bad = 0
for _ in range(20000):
    frames = rng.random(int(rng.integers(0, 300))) < rng.random()
    loud = old.runs(frames)
    for dip in (1, 2, 5, 20, 50):
        if not loud.size:
            ref = []
        else:
            opens = np.concatenate(([True], loud[1:, 0] - loud[:-1, 1] >= dip))
            closes = np.concatenate((opens[1:], [True]))
            ref = list(zip(loud[opens, 0].tolist(), loud[closes, 1].tolist()))
        got = [tuple(x) for x in new._joined(loud, dip).tolist()]
        bad += ref != got
print("_joined vs inline: 100000 cases, mismatches", bad)
print("_joined(empty):", new._joined(new.runs(np.zeros(0, bool)), 20).shape, new._joined(new.runs(np.zeros(5, bool)), 20).shape)
