import sys
import numpy as np
import lib
SR = lib.SR
LA, LB, floor, el, gap, n, bgap, blen, tail, at_end = -30, -44, -66, 0.06, 0.35, 4, 0.45, 1.0, 0.6, True
rng = np.random.default_rng(3)
first = lib.turn(30, -20, rng)
a0, a1 = 0.5, 2.0
t0 = a1 + 2.9
train_end = t0 + n * el + (n - 1) * gap
b0 = train_end + bgap; b1 = b0 + blen; P = b1 + tail
pause = lib.noise(P, floor, rng)
lib.add(pause, a0, a1 - a0, LA, rng)
for k in range(n):
    lib.add(pause, t0 + k * (el + gap), el, LA - 4, rng)
lib.add(pause, b0, blen, LB, rng)
wav = np.concatenate([first, pause] + ([] if at_end else [lib.turn(30, -20, rng)]))
off = first.size / SR
print("A", off + a0, off + a1, "train", off + t0, off + train_end, "B", off + b0, off + b1, "end", wav.size / SR)
c = lib.CM
rms = c.frame_rms(wav)
g = c.relative_gate(rms, 0.4)
print("file gate dB", 20 * np.log10(g))
loud0 = rms > g
st = c._runs_of(~loud0, 150)
print("stretches", [(a * 0.02, b * 0.02) for a, b in st])
for a, b in st:
    part = rms[a:b]
    gate = max(c.relative_gate(part, 0.4), float(np.percentile(part, 10)) * c._OVER_FLOOR)
    print(" stretch gate dB", 20 * np.log10(gate))
for tag in ("main", "pr"):
    print(tag, lib.loud_runs_sec(tag, wav, off - 0.5, wav.size / SR))
