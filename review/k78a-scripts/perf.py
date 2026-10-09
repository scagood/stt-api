import sys, time; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/k78a-scripts")
from lib import *
rng = np.random.default_rng(3)
parts = []
tot = 0
while tot < 7200 * SR:
    dur = float(rng.choice([0.1, 0.3, 0.6, 2, 5, 20])); lvl = float(rng.uniform(-62, -15))
    n = int(dur * SR); parts.append((rng.standard_normal(n) * 10 ** (lvl / 20)).astype(np.float32)); tot += n
wav = np.concatenate(parts)
for n in ("main", "head"):
    ch, rt = MODS[n]
    best = 1e9
    for _ in range(3):
        t = time.perf_counter(); s = ch._volume_speech_segments(wav); p = rt.pauses(wav); best = min(best, time.perf_counter() - t)
    print(n, f"{wav.size/SR/3600:.2f} h: VAD+pauses {best:.2f} s, {len(s)} segments, {len(p)} pauses")
