import sys, time; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/m75b-scripts")
from lib import *
FR = 50; H = 3600 * FR
rng = np.random.default_rng(0)
floor = 10 ** (-55 / 20)
def mk(pattern_fn, interleave):
    r = floor * rng.uniform(0.9, 1.1, 2 * H)
    if interleave:  # 60 s loud / 60 s pattern
        for k in range(0, 2 * H, 120 * FR): r[k: k + 60 * FR] = 0.1
    else:
        r[:H] = 0.1
    pattern_fn(r)
    if interleave:
        for k in range(0, 2 * H, 120 * FR): r[k: k + 60 * FR] = 0.1
    else:
        r[:H] = 0.1
    return r.astype(np.float32)
def bursts(r):  # 3-8 frame bursts every 0.5-3 s
    k = 0
    while k < r.size:
        L = int(rng.integers(3, 9)); r[k: k + L] = 10 ** (-40 / 20); k += L + int(rng.integers(25, 150))
def dense(r):  # 6-frame bursts every 18-25 frames: joined into long sounds
    k = 0
    while k < r.size:
        r[k: k + 6] = 10 ** (-40 / 20); k += 6 + int(rng.integers(12, 19))
def sylls(r):  # 3 on 3 off syllables in 2 s phrases, 2 s gaps
    idx = np.arange(r.size); on = ((idx // 3) % 2 == 0) & ((idx // 100) % 2 == 0); r[on] = 10 ** (-40 / 20)
def levels(r):  # 0.6 s sounds every 4 s, random level
    for k in range(0, r.size, 200): r[k: k + 30] = 10 ** (rng.uniform(-50, -30) / 20)
def peel(r):  # 0.6 s sounds every 3.2 s with steadily falling level: each pass can find one tier
    lv = np.linspace(-30, -54, r.size // 160 + 1)
    for j, k in enumerate(range(0, r.size, 160)): r[k: k + 30] = 10 ** (lv[j] / 20)
def short_words(r):  # 0.3 s sounds every 0.9 s, 1 s sound every 20 s
    for k in range(0, r.size, 45): r[k: k + 15] = 10 ** (-40 / 20)
    for k in range(0, r.size, 1000): r[k: k + 50] = 10 ** (-40 / 20)
for name, fn in [("bursts", bursts), ("dense", dense), ("sylls", sylls), ("levels", levels), ("peel", peel), ("short_words", short_words)]:
    for inter in (False, True):
        rms = mk(fn, inter)
        out = [f"{name}{'/interleaved' if inter else '/1h-run'}"]
        for n in ("main", "prev", "head"):
            for ratio in (0.4, 0.6):
                t = time.perf_counter(); m = loudmask(n, rms, ratio, 150); dt = time.perf_counter() - t
                out.append(f"{n}@{ratio}:{dt:.2f}s/{m[H:].mean() if not inter else m.mean():.3f}")
        print("  ".join(out), flush=True)
