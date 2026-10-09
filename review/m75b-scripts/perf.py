import sys, time; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/m75b-scripts")
from lib import *
ch_h = MODS["head"][0]
FR = 50  # frames/s
H2 = 2 * 3600 * FR
rng = np.random.default_rng(0)
floor = 10 ** (-55 / 20)
def frames_from(levels_db_fn):
    return levels_db_fn()
cases = {}
# a) 2 h loud narration with 1 s pauses every 10 s and 5 s pauses every 60 s (room tone)
r = np.full(H2, 10 ** (-20 / 20)) * rng.uniform(0.5, 1.5, H2)
for k in range(0, H2, 10 * FR): r[k: k + FR] = floor * rng.uniform(0.9, 1.1, FR)
for k in range(0, H2, 60 * FR): r[k: k + 5 * FR] = floor * rng.uniform(0.9, 1.1, 5 * FR)
cases["narration"] = r
# b) 1 min loud, then 2 h quiet speaker (-46) with syllable fragments: alternating 3 on / 3 off frames, phrase gaps
r = floor * rng.uniform(0.9, 1.1, H2)
r[: 60 * FR] = 0.1
idx = np.arange(H2)
on = ((idx // 3) % 2 == 0) & ((idx // 100) % 2 == 0) & (idx >= 60 * FR)
r[on] = 10 ** (-40 / 20)
cases["quiet-speaker-2h"] = r
# c) 1 min loud then 2 h of room tone with random 3-8 frame bursts every 0.5-3 s (many short heard runs, none >= 25)
r = floor * rng.uniform(0.9, 1.1, H2); r[: 60 * FR] = 0.1
k = 60 * FR
while k < H2:
    L = int(rng.integers(3, 9)); r[k: k + L] = 10 ** (-40 / 20); k += L + int(rng.integers(25, 150))
cases["bursts-2h"] = r
# d) 1 min loud then dense: bursts 6 frames, every 25-30 frames (joined -> long sounds -> near everywhere)
r = floor * rng.uniform(0.9, 1.1, H2); r[: 60 * FR] = 0.1
k = 60 * FR
while k < H2:
    r[k: k + 6] = 10 ** (-40 / 20); k += 6 + int(rng.integers(12, 19))
cases["dense-joined-2h"] = r
# e) peeling: levels decreasing — 1 min loud then 2h with sounds of 0.6 s at random levels spread every 4 s
r = floor * rng.uniform(0.9, 1.1, H2); r[: 60 * FR] = 0.1
for j, k in enumerate(range(60 * FR, H2, 200)):
    r[k: k + 30] = 10 ** (rng.uniform(-50, -30) / 20)
cases["sounds-every-4s-2h"] = r
for name, rms in cases.items():
    rms = rms.astype(np.float32)
    out = [name]
    for n in ("main", "prev", "head"):
        for ratio in (0.4, 0.6):
            t = time.perf_counter(); m = loudmask(n, rms, ratio, 150); dt = time.perf_counter() - t
            out.append(f"{n}@{ratio}: {dt:.2f}s loud={m.mean():.3f}")
    print("  ".join(out))
