"""Rebuild scene i of fuzz.py (seed, ms) and show the region around time T."""
import sys
import numpy as np
ARGS = sys.argv[1:]  # SEED MS IDX T
sys.argv = [sys.argv[0], "0", ARGS[0], ARGS[1]] + ARGS[4:]
import fuzz  # runs nothing with N=0
import lib
idx, T = int(ARGS[2]), float(ARGS[3])
rng = np.random.default_rng(fuzz.SEED)
for i in range(idx + 1):
    wav, speech, nonspeech = fuzz.scene(rng)
lo, hi = T - 8, T + 8
print("speech", [(round(a, 2), round(b, 2)) for a, b in speech if b > lo and a < hi])
print("nonspeech", [(round(a, 2), round(b, 2)) for a, b in nonspeech if b > lo and a < hi])
for tag in ("main", "pr"):
    pl = lib.plan(tag, wav)
    print(tag, "loud", lib.loud_runs_sec(tag, wav, lo, hi))
    print(tag, "segs", [s for s in lib.sec(pl.speech) if s[1] > lo and s[0] < hi])
    print(tag, "ranges", [s for s in lib.sec(pl.ranges) if s[1] > lo - 20 and s[0] < hi + 20])
    print(tag, "retime loud", lib.loud_runs_sec(tag, wav, lo, hi, ratio=0.6))
