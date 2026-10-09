import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/k78a-scripts")
from lib import *
cm, rm = MODS["main"]; ch, rh = MODS["head"]
seed = int(sys.argv[1]); target = int(sys.argv[2]); ratio = float(sys.argv[3])
rng = np.random.default_rng(seed)
for it in range(target + 1):
    floor = float(rng.uniform(-62, -50))
    P = float(rng.uniform(6, 30))
    sounds = []
    t = 40 + float(rng.uniform(0.3, 3))
    ph_end = None
    for _ in range(int(rng.integers(1, 4))):
        d = float(rng.uniform(0.5, 3.0)); lvl = float(rng.uniform(-36, -31))
        if t + d > 40 + P - 0.5: break
        sounds.append((t, d, lvl)); ph_end = t + d
        t += d + float(rng.uniform(0.1, 1.0))
    if ph_end is None: continue
    rel = 3.0
    w0 = ph_end + rel - float(rng.uniform(0.02, 0.4)); wd = float(rng.uniform(0.1, 0.48))
    sounds.append((w0, wd, float(rng.uniform(-40, -31))))
    t = w0 + wd + float(rng.uniform(0.05, 1.0))
    while t < 40 + P - 0.2:
        d = float(rng.uniform(0.1, 1.2)); lvl = floor + float(rng.uniform(6, 16))
        d = min(d, 40 + P - t)
        sounds.append((t, d, lvl))
        t += d + float(rng.uniform(0.05, 2.5))
    if it == target:
        wav = build(P, sounds, seed=it + 100000 * seed, floor_db=floor)
print("floor", round(floor,1), "pause 40 ..", round(40+P,2))
for s in sounds: print("  sound %.2f-%.2f  %.1f dBFS (%.1f over floor)" % (s[0], s[0]+s[1], s[2], s[2]-floor))
rms = cm.frame_rms(wav)
a = cm.loud_frames(rms, ratio, 150); b = ch.loud_frames(rms, ratio, 150)
def r(m): return [(round(x*0.02,2), round(y*0.02,2)) for x, y in runs(m).tolist()]
from importlib import import_module
runs = cm.runs
print("main loud:", [x for x in r(a) if x[1] > 39.9 and x[0] < 40+P+0.1])
print("head loud:", [x for x in r(b) if x[1] > 39.9 and x[0] < 40+P+0.1])
print("main&~head:", r(a & ~b))
print("head&~main:", r(b & ~a))
if ratio == 0.6:
    print("main pauses:", [(round(x,2), round(y,2)) for x, y in rm.pauses(wav) if y > 40 and x < 40+P])
    print("head pauses:", [(round(x,2), round(y,2)) for x, y in rh.pauses(wav) if y > 40 and x < 40+P])
else:
    for n in ("main", "head"):
        print(n, "plan ctx0:", secs(plan(n, wav, 0.0).ranges))
