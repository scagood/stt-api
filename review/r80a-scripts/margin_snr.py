"""Quiet word X dB over the margin's floor (room tone or a noise bed): kept by main vs head? Real volume VAD, v2 ctx5.
usage: margin_snr.py GEN N FLOOR(room|noise50|noise45)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from common import MAIN, HEAD, SR, plan
from gens import gen_a, gen_b, quiet_word, pink

gen, N, floor_kind = sys.argv[1], int(sys.argv[2]), sys.argv[3]
G = gen_a if gen == "a" else gen_b
room = -65.0 if gen == "a" else -62.0
for side in ("tail", "lead"):
    for X in (3, 5, 7, 9, 11, 13, 16, 20):
        T = Counter()
        for seed in range(N):
            r = np.random.default_rng([seed, X, 1 if side == "tail" else 2, ord(gen)])
            total = 40 * SR
            wav, words = G([(3 * SR, 37 * SR)], total, r)
            fl = room
            if floor_kind.startswith("noise"):
                fl = -float(floor_kind[5:])
                wav += (pink(total, r) if gen == "b" else r.standard_normal(total).astype(np.float32)) * 10 ** (fl / 20)
            w = (38.3, 38.7) if side == "tail" else (1.4, 1.8)
            a, b = int(w[0] * SR), int(w[1] * SR)
            # word level: X dB over the floor (power sum)
            quiet_word(wav, a, b, fl + X, r, gen)
            res = {}
            for name, mod in (("main", MAIN), ("head", HEAD)):
                p = plan(mod, wav, "v2 ctx5", gate=None)
                res[name] = (any(r0 <= a and b <= r1 for r0, r1 in p.ranges), any(s < b and e > a for s, e in p.speech))
            T["main"] += res["main"][0]; T["head"] += res["head"][0]; T["vad"] += res["main"][1]
            T["fewer"] += res["main"][0] and not res["head"][0]
        print(f"gen {gen} floor {floor_kind} ({fl:.0f} dBFS) {side} word +{X} dB over floor: VAD heard {T['vad']}/{N}, in a range main {T['main']} head {T['head']}, head fewer {T['fewer']}", flush=True)
