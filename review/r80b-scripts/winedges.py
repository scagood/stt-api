"""Context-window edges that land inside a word (the #69 early-stop exposure): main vs head.
usage: winedges.py SEEDS N GAP_DB [scales]"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from synth import MAIN, PR, SR, CONFIGS, layout, synth, plan, in_word

seeds = [int(x) for x in sys.argv[1].split(",")]
N, gap_db = int(sys.argv[2]), float(sys.argv[3])
scales = tuple(float(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 else (1.0, 1.6)
for cname in ("v2 ctx5", "v3 ctx5"):
    cfg = CONFIGS[cname]
    T = Counter()
    for seed in seeds:
        for scale in scales:
            rng = random.Random(f"we|{seed}|{cname}|{scale}")
            nrng = np.random.default_rng(seed * 1000 + int(scale * 10))
            for _ in range(N):
                segs_s, total_s = layout(rng, scale)
                if total_s <= cfg["mx"]:
                    continue
                total = int(round(total_s * SR)); segs = [(int(round(a * SR)), int(round(b * SR))) for a, b in segs_s]
                wav, words = synth(segs, total, nrng, gap_db, (50, 350))
                starts = np.array([w[0] for w in words])
                res = {}
                for name, mod in (("main", MAIN), ("head", PR)):
                    r, w = plan(mod, wav, segs, cfg)
                    edges = []
                    for i, ((a, b), (wa, wb)) in enumerate(zip(r, w)):
                        if i and r[i - 1][1] == a and wa < a:
                            edges.append(wa)
                        if i + 1 < len(r) and r[i + 1][0] == b and wb > b:
                            edges.append(wb)
                    inspeech = [e for e in edges if any(s < e < t for s, t in segs)]
                    res[name] = (len(edges), len(inspeech), sum(in_word(e, starts, words) for e in edges))
                T["layouts"] += 1
                for name in res:
                    T[name + "_edges"] += res[name][0]; T[name + "_in_speech"] += res[name][1]; T[name + "_in_word"] += res[name][2]
                T["head_more_in_word"] += res["head"][2] > res["main"][2]
                T["head_fewer_in_word"] += res["head"][2] < res["main"][2]
    f = lambda k: T[k]
    print(f"{cname} gaps {gap_db} dB: layouts {f('layouts')}; context edges main {f('main_edges')} (in VAD speech {f('main_in_speech')}, in a word {f('main_in_word')}) "
          f"-> head {f('head_edges')} (in VAD speech {f('head_in_speech')}, in a word {f('head_in_word')}); layouts with more edges in words {f('head_more_in_word')}, fewer {f('head_fewer_in_word')}", flush=True)
