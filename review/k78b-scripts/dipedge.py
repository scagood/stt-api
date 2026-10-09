"""Two 0.3 s breaths (-42 over -55) alone in a 10 s pause, gap g apart: decoded (joined into one
0.6 s sound) or cut out? README: 'breaths 400 ms or more apart ... never add up to speech'."""
import numpy as np
import lib
SR = lib.SR
for g in (0.38, 0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46):
    out = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        first = lib.turn(40, -20, rng)
        pause = lib.noise(10, -55, rng)
        at = 4.5 + rng.uniform(0, 0.02)
        lib.add(pause, at, 0.3, -42, rng)
        lib.add(pause, at + 0.3 + g, 0.3, -42, rng)
        wav = np.concatenate([first, pause, lib.turn(40, -20, rng)])
        rg = lib.plan("pr", wav).ranges
        out.append(lib.covered(rg, int(44 * SR), int(46 * SR)) > 0)
    print(f"gap {g:.2f} s: decoded in {sum(out)}/20 seeds")
