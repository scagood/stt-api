import random, sys, os
from collections import Counter
from multiprocessing import Pool
from h import *
from fz import gen
KEYS = ("n", "isc", "silent", "short", "lost", "wedge")
def work(args):
    cname, g, scale, seed, N = args
    cfg = CONFIGS[cname]; import zlib; rng = random.Random(zlib.crc32(repr((cname, g, scale, seed, "c3")).encode()))
    T = {k: Counter() for k in MODS}; C = Counter(); ex = {}
    for i in range(N):
        segs, total = gen(rng, cfg, scale, g)
        if total <= cfg["mx"]: continue
        m = {}
        for k, mod in MODS.items():
            r, w, sp, t = run(mod, segs, total, cfg)
            m[k] = metrics(r, w, sp, t, cfg); m[k]["r"] = r
            for key in KEYS: T[k][key] += m[k][key]
        T["main"]["layouts"] += 1
        for k in MODS:
            if k == "main": continue
            for key in ("n", "isc", "silent", "short", "lost"):
                if m[k][key] > m["main"][key]:
                    C[k + " " + key + "_worse"] += 1
                    if (k, key) not in ex or len(segs) < len(ex[(k, key)][0]): ex[(k, key)] = (segs, total, sec(m["main"]["r"]), sec(m[k]["r"]))
    return cname, {k: dict(v) for k, v in T.items()}, dict(C), ex
if __name__ == "__main__":
    N = int(sys.argv[1]); seeds = int(sys.argv[2]); gens = sys.argv[3].split(","); scales = [float(x) for x in sys.argv[4].split(",")]
    cfgs = sys.argv[5].split(",") if len(sys.argv) > 5 else list(CONFIGS)
    jobs = [(c, g, s, sd, N) for c in cfgs for g in gens for s in scales for sd in range(seeds)]
    agg = {}
    with Pool(4) as p:
        for c, T, C, ex in p.imap_unordered(work, jobs):
            A = agg.setdefault(c, [{k: Counter() for k in MODS}, Counter(), {}])
            for k in MODS: A[0][k].update(T[k])
            A[1].update(C)
            for k, e in ex.items():
                if k not in A[2] or len(e[0]) < len(A[2][k][0]): A[2][k] = e
    for c, (T, C, ex) in agg.items():
        print(c, "layouts", T["main"]["layouts"])
        for k in MODS: print("   ", k, {key: T[k][key] for key in KEYS}, "isc vs main %.1f%%" % (100 * (T[k]["isc"] - T["main"]["isc"]) / max(1, T["main"]["isc"])))
        print("   worse vs main:", dict(sorted(C.items())))
        for k, e in ex.items(): print("   EX", k, e)
