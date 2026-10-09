"""PARAKEET_VAD_MIN_SILENCE_MS above 400: a quiet halting reply, four 0.3 s words 0.6 s apart
(-40 dBFS over -55 dBFS room tone), alone in a 14 s pause between two 40 s turns at -20 dBFS.
Args: MS [gap]"""
import sys
import numpy as np
import lib
SR = lib.SR
MS = int(sys.argv[1]); GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
for c in (lib.CM, lib.CP):
    c.VAD_MIN_SILENCE_MS = MS
rng = np.random.default_rng(1)
first = lib.turn(40, -20, rng)
pause = lib.noise(14, -55, rng)
words = [(4.0 + k * (0.3 + GAP), 0.3) for k in range(4)]
for at, ln in words:
    lib.add(pause, at, ln, -40, rng)
wav = np.concatenate([first, pause, lib.turn(40, -20, rng)])
print(f"min_silence {MS} ms; words at", [(40 + a, round(40 + a + l, 2)) for a, l in words])
for tag in ("main", "pr"):
    pl = lib.plan(tag, wav)
    got = sum(lib.covered(pl.ranges, int((40 + a) * SR), int((40 + a + l) * SR)) for a, l in words) / SR
    print(f"  {tag}: ranges {lib.sec(pl.ranges)}  word audio decoded {got:.2f} of {sum(l for _, l in words):.2f} s")
    print(f"  {tag}: retime pauses over the words", [(round(a, 2), round(b, 2)) for a, b in lib.pauses(tag, wav) if a < 40 + words[-1][0] + 0.3 and b > 40 + words[0][0]])
