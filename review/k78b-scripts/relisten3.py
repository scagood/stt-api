"""Smallest level gap between quiet phrase A and later quieter phrase B for which retime.pauses
(0.6) on the PR loses B that main keeps, per A length. Floor -64, train 4 x 60 ms clicks 0.3 s
apart starting 2.9 s after A, B 0.8 s starting 0.45 s after the train, 1.2 s more pause,
then a loud turn."""
import numpy as np
import lib
SR = lib.SR
for ALEN in (1.0, 2.0, 3.0, 5.0, 8.0):
    for LA in (-33, -36, -40):
        found = []
        for dB_B in range(4, 21):
            LB = LA - dB_B
            if LB < -64 + 11:
                break
            rng = np.random.default_rng(3)
            first = lib.turn(30, -20, rng)
            a0, a1 = 0.5, 0.5 + ALEN
            t0 = a1 + 2.9; n, el, gap = 4, 0.06, 0.3
            te = t0 + n * el + (n - 1) * gap
            b0 = te + 0.45; b1 = b0 + 0.8; P = b1 + 1.2
            pause = lib.noise(P, -64, rng)
            pause[int(a0 * SR):int(a1 * SR)] += lib.turn(ALEN, LA, rng)[: int(a1 * SR) - int(a0 * SR)]
            for k in range(n):
                lib.add(pause, t0 + k * (el + gap), el, LA - 4, rng)
            lib.add(pause, b0, 0.8, LB, rng)
            wav = np.concatenate([first, pause, lib.turn(30, -20, rng)])
            off = first.size / SR
            i0, i1 = int((off + b0) / 0.02) + 1, int((off + b1) / 0.02) - 1
            m = int(lib.loud("main", wav, ratio=0.6)[i0:i1].sum()); p = int(lib.loud("pr", wav, ratio=0.6)[i0:i1].sum())
            ms = int(lib.loud("main", wav, ratio=0.4)[i0:i1].sum()); ps = int(lib.loud("pr", wav, ratio=0.4)[i0:i1].sum())
            if m > p or ms > ps:
                found.append((dB_B, (m, p), (ms, ps)))
        print(f"A {ALEN} s at {LA} dBFS: B lost on PR at A-B gaps {found}")
