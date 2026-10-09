"""One straddling-train case: 0.06 s clicks every 0.36 s x7 starting 2.8 s after a quiet 1 s phrase."""
import sys
import numpy as np
import lib
SR = lib.SR
el, gap, n, lvl, P = 0.06, 0.3, 7, -44, 10.0
rng = np.random.default_rng(1)
first = lib.turn(40, -20, rng)
pause = lib.noise(P, -55, rng)
lib.add(pause, 1.0, 1.0, -40, rng)
for i in range(n):
    lib.add(pause, 4.8 + i * (el + gap), el, lvl, rng)
wav = np.concatenate([first, pause, lib.turn(40, -20, rng)])
print("phrase 41.0-42.0, reach to 45.0; clicks", [round(40 + 4.8 + i * (el + gap), 2) for i in range(n)])
for tag in ("main", "pr"):
    pl = lib.plan(tag, wav)
    print(tag, "loud runs in pause", lib.loud_runs_sec(tag, wav, 40.2, 49.8))
    print(tag, "segments", [s for s in lib.sec(pl.speech) if 39 < s[1] and s[0] < 51])
    print(tag, "ranges", lib.sec(pl.ranges))
    print(tag, "retime pauses 40-50", [(round(a, 2), round(b, 2)) for a, b in lib.pauses(tag, wav) if 40 < b and a < 50])
