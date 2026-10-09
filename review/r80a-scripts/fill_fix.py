import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from common import MAIN, HEAD, SR, plan, load_local
from gens import gen_a
FIX = load_local(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunker_fixA.py"), "fixA")
N = int(sys.argv[1])
for noise in (None, -50.0):
    for lq, gdb in ((-34, 6), (-34, 10), (-40, 6), (-40, 20)):
        T = Counter()
        for seed in range(N):
            r = np.random.default_rng([seed, int(-lq), gdb, 1])
            total = 40 * SR
            wav, words = gen_a([(3 * SR, 37 * SR)], total, r)
            Q = r.uniform(1.0, 2.7)
            qs = (0, int(Q * SR))
            q, qw = gen_a([qs], total, r, gap_db=gdb, base_db=lq, room_db=-120)
            m = np.zeros(total, bool); m[qs[0]:qs[1]] = True
            wav[m] = wav[m] * 0 + q[m] + (r.standard_normal(m.sum()).astype(np.float32) * 10 ** (-65 / 20))
            if noise is not None:
                wav += r.standard_normal(total).astype(np.float32) * 10 ** (noise / 20)
            res = {}
            for nm, mod in (("m", MAIN), ("h", HEAD), ("f", FIX)):
                p = plan(mod, wav, "v2 ctx5", gate=None)
                cov = np.zeros(total, bool)
                for a, b in p.ranges: cov[a:b] = True
                res[nm] = sum(cov[a:b].all() for a, b in qw)
            T["w"] += len(qw)
            for k in "mhf": T[k] += res[k]
            T["hf"] += res["h"] < res["m"]; T["ff"] += res["f"] < res["m"]
        print(f"lead quiet talker {lq} dBFS gaps {gdb} noise {noise}: words {T['w']} main {T['m']} head {T['h']} fixA {T['f']}; files fewer than main: head {T['hf']}/{N} fixA {T['ff']}/{N}", flush=True)
