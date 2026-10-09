# Target the recursion: a straddling word (head keeps its tail) makes head's
# leftover quiet run shorter / different; can main then hear something in it
# that head doesn't?  Nested levels: a quiet speaker, and a much quieter one.
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/k78a-scripts")
from lib import *
cm, rm = MODS["main"]; ch, rh = MODS["head"]
seed = int(sys.argv[1]); N = int(sys.argv[2])
rng = np.random.default_rng(seed)
lost = []; gained = 0; checks = 0
for it in range(N):
    floor = float(rng.uniform(-62, -50))
    P = float(rng.uniform(6, 30))
    sounds = []
    # anchor phrase(s), fairly loud (but under the file gate ~ -30)
    t = 40 + float(rng.uniform(0.3, 3))
    ph_end = None
    for _ in range(int(rng.integers(1, 4))):
        d = float(rng.uniform(0.5, 3.0)); lvl = float(rng.uniform(-36, -31))
        if t + d > 40 + P - 0.5: break
        sounds.append((t, d, lvl)); ph_end = t + d
        t += d + float(rng.uniform(0.1, 1.0))
    if ph_end is None: continue
    # straddling word near ph_end + relisten (3 s)
    rel = 3.0
    w0 = ph_end + rel - float(rng.uniform(0.02, 0.4)); wd = float(rng.uniform(0.1, 0.48))
    sounds.append((w0, wd, float(rng.uniform(-40, -31))))
    # quieter stuff after
    t = w0 + wd + float(rng.uniform(0.05, 1.0))
    while t < 40 + P - 0.2:
        d = float(rng.uniform(0.1, 1.2)); lvl = floor + float(rng.uniform(6, 16))
        d = min(d, 40 + P - t)
        sounds.append((t, d, lvl))
        t += d + float(rng.uniform(0.05, 2.5))
    wav = build(P, sounds, seed=it + 100000 * seed, floor_db=floor)
    rms = cm.frame_rms(wav)
    for ratio, rel in [(0.4, 150), (0.6, 150)]:
        checks += 1
        a = cm.loud_frames(rms, ratio, rel); b = ch.loud_frames(rms, ratio, rel)
        l = int((a & ~b).sum()); g = int((b & ~a).sum())
        if g: gained += 1
        if l: lost.append((l, it, ratio))
print(f"adversarial seed {seed}: {checks} checks, head gains in {gained}, head loses in {len(lost)}: {sorted(lost, reverse=True)[:10]}")
