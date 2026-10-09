"""Does keeping a straddling train whole stop the re-listen of what follows it?
Quiet stretch between two loud turns: phrase A (level LA) 1.5 s, a short click train starting
inside A's 3 s reach, then a quieter phrase B (level LB) shortly after the train, then the next
loud turn (or the file end). Search over levels/positions; report speech B frames that main
marks loud and PR does not, for the splitter (0.4) and retime (0.6)."""
import itertools

import numpy as np

import lib

SR = lib.SR
hits = []
for LA, dB_B, floor, el, gap, n, bgap, blen, tail, at_end in itertools.product(
    (-30, -34, -38), (6, 10, 14), (-66, -60), (0.06, 0.1), (0.25, 0.35), (2, 4, 6), (0.45, 0.8, 1.2), (0.6, 1.0), (0.2, 0.6, 1.2), (False, True)
):
    LB = LA - dB_B
    if LB < floor + 11:
        continue
    rng = np.random.default_rng(3)
    first = lib.turn(30, -20, rng)
    a0, a1 = 0.5, 2.0
    t0 = a1 + 2.9
    train_end = t0 + n * el + (n - 1) * gap
    b0 = train_end + bgap
    b1 = b0 + blen
    P = b1 + tail
    pause = lib.noise(P, floor, rng)
    lib.add(pause, a0, a1 - a0, LA, rng)
    for k in range(n):
        lib.add(pause, t0 + k * (el + gap), el, LA - 4, rng)
    lib.add(pause, b0, blen, LB, rng)
    parts = [first, pause] + ([] if at_end else [lib.turn(30, -20, rng)])
    wav = np.concatenate(parts)
    off = first.size / SR
    i0, i1 = int((off + b0) / 0.02) + 1, int((off + b1) / 0.02) - 1
    res = {}
    for ratio in (0.4, 0.6):
        lm = lib.loud("main", wav, ratio=ratio)[i0:i1].sum()
        lp = lib.loud("pr", wav, ratio=ratio)[i0:i1].sum()
        res[ratio] = (int(lm), int(lp))
    if any(m > p for m, p in res.values()):
        hits.append(((LA, LB, floor, el, gap, n, bgap, blen, tail, at_end), res))
print("cases with B frames loud on main but not PR:", len(hits))
for h in hits[:30]:
    print(h)
