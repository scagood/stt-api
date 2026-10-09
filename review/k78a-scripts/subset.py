# Does head ever leave quiet a frame main keeps loud (at the default 400 ms)?
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/k78a-scripts")
from lib import *
cm, _ = MODS["main"]; ch, _ = MODS["head"]
seed = int(sys.argv[1]); N = int(sys.argv[2]); mode = sys.argv[3]
rng = np.random.default_rng(seed)
lost = []; gained = 0; diff = 0; checks = 0
for it in range(N):
    if mode == "multi":
        parts = []
        for _ in range(rng.integers(3, 30)):
            dur = float(rng.choice([0.01, 0.03, 0.1, 0.2, 0.3, 0.45, 0.6, 1.5, 4, 8, 20]))
            lvl = float(rng.uniform(-75, -10))
            parts.append(rng.standard_normal(int(dur * SR)) * 10 ** (lvl / 20))
        wav = np.concatenate(parts).astype(np.float32)
    else:
        # a pause between loud turns with random quiet phrases/words/breaths, floors vary
        P = float(rng.uniform(5, 40)); floor = float(rng.uniform(-62, -50))
        sounds = []; t = 40 + float(rng.uniform(0, 2))
        while t < 40 + P - 0.3:
            kind = rng.random()
            if kind < 0.25: d = float(rng.uniform(0.5, 1.5)); lvl = floor + float(rng.uniform(8, 25))
            elif kind < 0.65: d = float(rng.uniform(0.1, 0.45)); lvl = floor + float(rng.uniform(8, 25))
            else: d = float(rng.uniform(0.08, 0.45)); lvl = floor + float(rng.uniform(5, 17))
            d = min(d, 40 + P - t)
            sounds.append((t, d, lvl))
            t += d + float(rng.choice([rng.uniform(0.05, 0.45), rng.uniform(0.45, 1.5), rng.uniform(1.5, 5)]))
        wav = build(P, sounds, seed=it + 100000 * seed, floor_db=floor)
    rms = cm.frame_rms(wav)
    for ratio, rel in [(0.4, 150), (0.6, 150)]:
        checks += 1
        a = cm.loud_frames(rms, ratio, rel); b = ch.loud_frames(rms, ratio, rel)
        if not np.array_equal(a, b): diff += 1
        l = int((a & ~b).sum()); g = int((b & ~a).sum())
        if g: gained += 1
        if l: lost.append((l, it, ratio))
print(f"{mode} seed {seed}: {N} wavs, {checks} checks, differ {diff}, head gains frames in {gained}, head loses frames in {len(lost)}: {sorted(lost, reverse=True)[:10]}")
