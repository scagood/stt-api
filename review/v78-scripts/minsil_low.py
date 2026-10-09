"""PARAKEET_VAD_MIN_SILENCE_MS=100 (Silero's own default): 40 s turn at -20 dBFS, then room tone
-55 dBFS to the file end (14 s). A quiet 0.25 s word (-42) 2.3 s after the turn, and 9 s after
the turn five footsteps (0.15 s each, 0.15 s apart, -40 dBFS). Args: MS"""
import sys
import numpy as np
import lib_b as lib
SR = lib.SR
MS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
for c in (lib.CM, lib.CO, lib.CP):
    c.VAD_MIN_SILENCE_MS = MS
rng = np.random.default_rng(2)
first = lib.turn(40, -20, rng)
tail = lib.noise(14, -55, rng)
lib.add(tail, 2.3, 0.25, -42, rng)
steps = [(9.0 + k * 0.3, 0.15) for k in range(5)]
for at, ln in steps:
    lib.add(tail, at, ln, -40, rng)
wav = np.concatenate([first, tail])
print(f"min_silence {MS}: word 42.30-42.55, footsteps {40 + steps[0][0]:.2f}-{40 + steps[-1][0] + 0.15:.2f}, file end 54.0")
for tag in ("main", "old", "head"):
    pl = lib.plan(tag, wav)
    w = lib.covered(pl.ranges, int(42.3 * SR), int(42.55 * SR)) / SR
    print(f"  {tag}: segments {[s for s in lib.sec(pl.speech) if s[1] > 39]} ranges {lib.sec(pl.ranges)} word decoded {w:.2f}/0.25 s")
