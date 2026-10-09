import random, sys
from collections import Counter
from multiprocessing import Pool
from h import *
from fz import gen, gap, phrase
KEYS = ("n", "isc", "silent", "short", "lost", "bad", "wedge")
def layout(rng, cfg):
    own = own_of(cfg)
    f = rng.choice([0.0, rng.uniform(0, 3.2), 3.0, rng.uniform(2.5, 3.0), rng.uniform(0, 0.5)])
    a = rng.choice([rng.uniform(0.3, cfg["mn"]), rng.uniform(cfg["mn"] - 4, cfg["mn"]), rng.uniform(own - 3, own)])
    g1 = rng.choice([rng.uniform(0.02, 2.999), rng.uniform(2.5, 2.999), rng.uniform(0.02, 0.5)])
    s2 = f + a + g1
    k = rng.randint(2, 4)
    whole = rng.choice([k * own - rng.uniform(0, 3.5), k * own - rng.uniform(0, 0.5), k * own + rng.uniform(0, 0.5), rng.uniform(own, 4 * own)])
    e2 = f + whole
    if e2 <= s2 + 0.3: e2 = s2 + rng.uniform(own, 3 * own)
    segs = [(f, f + a), (s2, e2)]
    t = e2 + rng.choice([rng.uniform(0.02, 2.999), rng.uniform(2.5, 2.999), rng.uniform(0.02, 0.5), rng.uniform(3, 6)])
    for _ in range(rng.randint(0, 6)):
        L = phrase(rng, own, 1.0, rng.choice(["mix", "over", "short", "short"]))
        segs.append((t, t + L)); t += L + gap(rng, rng.choice(["hi", "lo", "mix"]))
    total = segs[-1][1] + rng.choice([0, rng.uniform(0, 1.5), rng.uniform(1, 6)])
    return [(round(x, 3), round(y, 3)) for x, y in segs], round(total, 3)
def work(args):
    cname, seed, N = args
    cfg = CONFIGS[cname]; import zlib; rng = random.Random(seed * 7919 + zlib.crc32(cname.encode()))
    C = Counter(); ex = {}
    for i in range(N):
        segs, total = layout(rng, cfg)
        if total <= cfg["mx"]: continue
        m = {}
        for k, mod in MODS.items():
            r, w, sp, t = run(mod, segs, total, cfg)
            m[k] = metrics(r, w, sp, t, cfg); m[k]["r"] = r
        C["layouts"] += 1
        a, b = m["pr"], m["main"]
        if a["r"] != b["r"]:
            C["differ"] += 1
            e2 = int(round(segs[1][1] * SR))
            ba = {x for r in a["r"] for x in r if x > e2}; bb = {x for r in b["r"] for x in r if x > e2}
            if ba != bb: C["differ_downstream"] += 1
        for key in KEYS:
            if a[key] > b[key]:
                C[key + "_worse"] += 1
                e = (len(segs), segs, total, sec(b["r"]), sec(a["r"]))
                if key not in ex or e[0] < ex[key][0]: ex[key] = e
            elif a[key] < b[key]: C[key + "_better"] += 1
    return cname, dict(C), ex
if __name__ == "__main__":
    N = int(sys.argv[1]); seeds = int(sys.argv[2])
    jobs = [(c, s, N) for c in CONFIGS for s in range(seeds)]
    agg = {}
    with Pool(4) as p:
        for c, C, ex in p.imap_unordered(work, jobs):
            A = agg.setdefault(c, [Counter(), {}]); A[0].update(C)
            for k, e in ex.items():
                if k not in A[1] or e[0] < A[1][k][0]: A[1][k] = e
    for c, (C, ex) in agg.items():
        print(c, dict(sorted(C.items())))
        for k, e in ex.items(): print("   EX", k, e)
