"""How far past a quiet phrase's 3 s reach does the PR keep sound (splitter gate), max over click
trains (length el, period el+gap, n clicks) whose first click is just inside the reach? And does
the 14 s pause then still get cut out?"""
import numpy as np
import lib
SR = lib.SR
best = []
for el in (0.05, 0.06, 0.07, 0.08, 0.1):
    for gap in (0.34, 0.36, 0.38, 0.39):
        for n in range(2, 13):
            rng = np.random.default_rng(4)
            first = lib.turn(40, -20, rng)
            pause = lib.noise(14, -55, rng)
            lib.add(pause, 1.0, 1.0, -40, rng)  # phrase 41-42, reach to 45.0
            for k in range(n):
                lib.add(pause, 4.9 + k * (el + gap), el, -42, rng)
            wav = np.concatenate([first, pause, lib.turn(40, -20, rng)])
            out = {}
            for tag in ("main", "pr"):
                lr = lib.loud_runs_sec(tag, wav, 42.5, 53.9)
                rg = lib.plan(tag, wav).ranges
                out[tag] = (max((b for a, b in lr), default=0), round(lib.covered(rg, 40 * SR, 54 * SR) / SR, 2))
            best.append((round(out["pr"][0] - 45.0, 2), el, gap, n, out))
best = [b for b in best if b[4]["main"] != b[4]["pr"]]
print("differing cases", len(best))
best.sort(reverse=True)
for b in best[:6]:
    print("PR keeps to %.2f s past the reach; click %.2f s, gap %.2f s, n %d; main (last loud, pause s decoded) %s, PR %s" % (b[0], b[1], b[2], b[3], b[4]["main"], b[4]["pr"]))
print("main last loud past reach in those:", sorted({round(b[4]["main"][0] - 45, 2) for b in best}))
