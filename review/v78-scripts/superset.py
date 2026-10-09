"""Item 5: is head's loud_frames a superset of main's? args: N SEED MS [kind]
kind: rms (direct random rms arrays), wav (pause scenes built as audio)."""
import sys
import numpy as np
import lib_b as lib
N, SEED, MS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
KIND = sys.argv[4] if len(sys.argv) > 4 else "rms"
CM, CH = lib.MODS["main"][0], lib.MODS["head"][0]
CM.VAD_MIN_SILENCE_MS = MS; CH.VAD_MIN_SILENCE_MS = MS
rng = np.random.default_rng(SEED)
def rand_rms():
    n = int(rng.integers(200, 6000))
    lv = np.empty(n)
    i = 0
    base = rng.uniform(-70, -45)
    while i < n:
        k = int(rng.choice([1, 2, 3, 5, 8, 12, 18, 22, 26, 30, 50, 100, 200, 500]))
        r = rng.random()
        if r < 0.45: l = base + rng.normal(0, 1.5)
        elif r < 0.75: l = base + rng.uniform(3, 25)
        elif r < 0.9: l = rng.uniform(-30, -10)
        else: base = rng.uniform(-70, -45); l = base
        lv[i:i + k] = l + rng.normal(0, 1.0, size=min(k, n - i))
        i += k
    return (10 ** (lv / 20)).astype(np.float32)
def rand_wav_rms():
    P = float(rng.uniform(5, 40)); floor = float(rng.uniform(-62, -50))
    first = lib.turn(float(rng.uniform(5, 40)), -20, rng)
    pause = lib.noise(P, floor, rng)
    t = float(rng.uniform(0, 2))
    while t < P - 0.3:
        k = rng.random()
        if k < 0.25: d = float(rng.uniform(0.5, 1.5)); l = floor + float(rng.uniform(8, 25))
        elif k < 0.65: d = float(rng.uniform(0.1, 0.45)); l = floor + float(rng.uniform(8, 25))
        else: d = float(rng.uniform(0.04, 0.45)); l = floor + float(rng.uniform(5, 17))
        lib.add(pause, t, min(d, P - t), l, rng)
        t += d + float(rng.choice([rng.uniform(0.05, 0.45), rng.uniform(0.45, 1.5), rng.uniform(1.5, 5)]))
    wav = np.concatenate([first, pause, lib.turn(float(rng.uniform(5, 40)), -20, rng)])
    return CM.frame_rms(wav)
viol = []; differ = 0; gains = 0; checks = 0; lostframes = 0
for it in range(N):
    rms = rand_rms() if KIND == "rms" else rand_wav_rms()
    for ratio, rel in ((0.4, 150), (0.6, 150), (0.4, 50)):
        checks += 1
        a = CM.loud_frames(rms, ratio, rel); b = CH.loud_frames(rms, ratio, rel)
        assert a.dtype == bool and b.shape == rms.shape
        if not np.array_equal(a, b): differ += 1
        if (b & ~a).any(): gains += 1
        l = int((a & ~b).sum())
        if l: viol.append((l, it, ratio, rel)); lostframes += l
print(f"{KIND} ms {MS} seed {SEED}: {N} arrays, {checks} checks, differ {differ}, head gains in {gains}, head loses frames in {len(viol)} (total {lostframes} frames): {sorted(viol, reverse=True)[:6]}")
