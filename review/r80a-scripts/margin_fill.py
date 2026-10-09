"""A margin filled with quieter speech VAD misses (a quieter talker, or a phrase cut by the file's start/end), v2 ctx5, real VAD.
Words of the quiet talker fully inside a range: main vs head. usage: margin_fill.py N"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from common import MAIN, HEAD, SR, plan
from gens import gen_a

N = int(sys.argv[1])
noise = float(sys.argv[2]) if len(sys.argv) > 2 else None
for side in ("lead", "tail"):
    for lq in (-34, -40, -46):
        for gdb in (6, 10, 20):
            T = Counter()
            for seed in range(N):
                r = np.random.default_rng([seed, int(-lq), gdb, 1 if side == "lead" else 2])
                total = 40 * SR
                wav, words = gen_a([(3 * SR, 37 * SR)], total, r)
                Q = r.uniform(1.0, 2.7)
                if side == "lead":
                    qs = (0, int(Q * SR))
                else:
                    qs = (total - int(Q * SR), total)
                q, qw = gen_a([qs], total, r, gap_db=gdb, base_db=lq, room_db=-120)
                m = np.zeros(total, bool); m[qs[0]:qs[1]] = True
                wav[m] = wav[m] * 0 + q[m] + (r.standard_normal(m.sum()).astype(np.float32) * 10 ** (-65 / 20))
                if noise is not None:
                    wav += r.standard_normal(total).astype(np.float32) * 10 ** (noise / 20)
                res = {}
                for nm, mod in (("m", MAIN), ("h", HEAD)):
                    p = plan(mod, wav, "v2 ctx5", gate=None)
                    cov = np.zeros(total, bool)
                    for a, b in p.ranges: cov[a:b] = True
                    res[nm] = sum(cov[a:b].all() for a, b in qw)
                    res[nm + "heard"] = sum(any(s < b and e > a for s, e in p.speech) for a, b in qw)
                T["words"] += len(qw); T["m"] += res["m"]; T["h"] += res["h"]; T["heard"] += res["mheard"]
                T["files_fewer"] += res["h"] < res["m"]; T["words_fewer"] += max(0, res["m"] - res["h"])
            print(f"{side} quiet talker {lq} dBFS gaps {gdb} dB down{'' if noise is None else f', noise {noise} dBFS'}: words {T['words']} (VAD heard {T['heard']}), "
                  f"covered main {T['m']} head {T['h']}; files where head covers fewer {T['files_fewer']}/{N} ({T['words_fewer']} words)", flush=True)
