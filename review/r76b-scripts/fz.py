import sys, random, json, time
from collections import Counter
from multiprocessing import Pool
from h import *
KEYS = ("n", "isc", "silent", "short", "lost", "bad", "wedge")
def gap(rng, kind):
    if kind == "hi":
        return rng.choice([rng.uniform(2.5, 2.999), rng.uniform(2.5, 2.999), rng.uniform(0.05, 2.5), rng.uniform(3, 8)])
    if kind == "lo":
        return rng.choice([0.0, rng.uniform(0, 0.5), rng.uniform(0, 0.5), rng.uniform(0.5, 2.9), rng.uniform(3, 6)])
    return rng.choice([rng.uniform(0.05, 1.0), rng.uniform(0.5, 2.9), rng.uniform(3, 8)])
def phrase(rng, own, scale, kind):
    if kind == "over":
        return rng.choice([own + rng.uniform(0, 6), own * 2 + rng.uniform(-3, 6), rng.uniform(0.3, 6), rng.uniform(5, own)]) * (scale if rng.random() < 0.3 else 1)
    if kind == "short":
        return rng.uniform(0.2, 6) * scale
    return rng.choice([rng.uniform(0.3, 6), rng.uniform(5, 25), rng.uniform(15, 50)]) * scale
def gen(rng, cfg, scale, g):
    own = own_of(cfg)
    if g == "mix":
        t = rng.uniform(0, 4); n = rng.randint(1, 8); gk = "mix"; pk = "mix"
    elif g == "gaphi":
        t = rng.uniform(0, 4); n = rng.randint(2, 10); gk = "hi"; pk = "mix"
    elif g == "gaplo":
        t = rng.uniform(0, 4); n = rng.randint(2, 12); gk = "lo"; pk = "mix"
    elif g == "many":
        t = rng.uniform(0, 4); n = rng.randint(8, 40); gk = rng.choice(["mix", "hi", "lo"]); pk = "short"
    elif g == "over":
        t = rng.uniform(0, 4); n = rng.randint(2, 8); gk = rng.choice(["mix", "hi", "lo"]); pk = "over"
    elif g == "lead":
        # first phrase with a lead margin, short; then long phrase; short pause; then random
        t = rng.uniform(0, 3.5)
        segs = []
        L = rng.uniform(0.3, own * 1.1); segs.append((t, t + L)); t += L + rng.uniform(0.05, 2.99)
        L = rng.uniform(own * 0.8, own * 3.2) * (scale if scale > 1 else 1); segs.append((t, t + L)); t += L + gap(rng, rng.choice(["hi", "lo", "mix"]))
        for _ in range(rng.randint(0, 6)):
            L = phrase(rng, own, scale, rng.choice(["mix", "over", "short"])); segs.append((t, t + L)); t += L + gap(rng, rng.choice(["hi", "lo", "mix"]))
        total = segs[-1][1] + rng.choice([0, rng.uniform(0, 1.5), rng.uniform(1, 6)])
        return [(round(a, 3), round(b, 3)) for a, b in segs], round(total, 3)
    elif g == "longend":
        t = rng.uniform(0, 4); n = rng.randint(1, 6); gk = rng.choice(["mix", "hi", "lo"]); pk = "mix"
        segs = []
        for _ in range(n):
            L = phrase(rng, own, scale, pk); segs.append((t, t + L)); t += L + gap(rng, gk)
        L = rng.uniform(own, own * 3.5); segs.append((t, t + L))
        total = segs[-1][1] + rng.choice([0, rng.uniform(0, 0.5), rng.uniform(0, 3), rng.uniform(3, 6)])
        return [(round(a, 3), round(b, 3)) for a, b in segs], round(total, 3)
    elif g == "grid":
        q = rng.choice([0.5, 0.25, own / 4, 1.0])
        t = q * rng.randint(0, 6); n = rng.randint(1, 8); segs = []
        for _ in range(n):
            L = q * rng.randint(1, int(3 * own / q)); segs.append((t, t + L)); t += L + q * rng.randint(0, int(3 / q) + 2)
        total = segs[-1][1] + q * rng.randint(0, 8)
        return [(round(a, 4), round(b, 4)) for a, b in segs], round(total, 4)
    elif g == "degen":
        k = rng.choice(["one", "touch", "zero", "shorttot", "overlap", "beyond"])
        if k == "one":
            a = rng.uniform(0, 5); L = rng.uniform(0.1, own * 4); return [(a, a + L)], a + L + rng.choice([0, rng.uniform(0, 5)])
        if k in ("touch", "zero"):
            t = rng.uniform(0, 3); segs = []
            for _ in range(rng.randint(2, 8)):
                L = phrase(rng, own, scale, "mix"); segs.append((round(t, 3), round(t + L, 3))); t += L + (0 if k == "touch" else rng.choice([0, 0.001, 0.01]))
            return segs, round(t + rng.uniform(0, 3), 3)
        if k == "shorttot":
            T = rng.uniform(0.5, cfg["mx"] + 2); t = 0; segs = []
            while t < T:
                L = rng.uniform(0.2, 10); segs.append((round(t, 3), round(min(T, t + L), 3))); t += L + rng.uniform(0, 4)
            return segs, round(T, 3)
        if k == "overlap":
            t = rng.uniform(0, 3); segs = []
            for _ in range(rng.randint(2, 8)):
                L = phrase(rng, own, scale, "mix"); segs.append((round(t, 3), round(t + L, 3))); t += L - rng.uniform(-2, 3)
            return segs, round(max(b for a, b in segs) + rng.uniform(0, 3), 3)
        # beyond: segments past total
        t = rng.uniform(0, 3); segs = []
        for _ in range(rng.randint(1, 6)):
            L = phrase(rng, own, scale, "mix"); segs.append((round(t, 3), round(t + L, 3))); t += L + gap(rng, "mix")
        return segs, round(segs[-1][1] - rng.uniform(0, 10), 3)
    segs = []
    for _ in range(n):
        L = phrase(rng, own, scale, pk); segs.append((t, t + L)); t += L + gap(rng, gk)
    total = segs[-1][1] + rng.choice([0, rng.uniform(0, 1.5), rng.uniform(1, 6)])
    return [(round(a, 3), round(b, 3)) for a, b in segs], round(total, 3)
def work(args):
    cname, g, scale, seed, N = args
    cfg = CONFIGS[cname]
    import zlib; rng = random.Random(zlib.crc32(repr((cname, g, scale, seed)).encode()))
    tot = {k: Counter() for k in MODS}; cmp = Counter(); ex = {}
    diff = 0
    for i in range(N):
        segs, total = gen(rng, cfg, scale, g)
        if total <= 0: continue
        m = {}
        for k, mod in MODS.items():
            r, w, sp, t = run(mod, segs, total, cfg)
            m[k] = metrics(r, w, sp, t, cfg); m[k]["r"] = r
        tot["main"]["layouts"] += 1
        for k in MODS:
            for key in KEYS: tot[k][key] += m[k][key]
            tot[k]["sil"] += m[k]["sil"]
        a, b = m["pr"], m["main"]
        if a["r"] != b["r"]: cmp["differ"] += 1
        for key in KEYS:
            if a[key] > b[key]:
                cmp[key + "_worse"] += 1
                e = (len(segs), segs, total, sec(b["r"]), sec(a["r"]), b[key], a[key])
                if key not in ex or e[0] < ex[key][0]: ex[key] = e
            elif a[key] < b[key]: cmp[key + "_better"] += 1
        if a["sil"] > b["sil"] + 1: cmp["sil_more"] += 1
        if a["sil"] + 1 < b["sil"]: cmp["sil_less"] += 1
    return cname, g, scale, {k: dict(v) for k, v in tot.items()}, dict(cmp), ex
if __name__ == "__main__":
    N = int(sys.argv[1]); seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    gens = sys.argv[3].split(",") if len(sys.argv) > 3 else ["mix", "gaphi", "gaplo", "many", "over", "lead", "longend", "grid", "degen"]
    scales = [float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0.5, 1.0, 1.6, 2.5]
    cfgs = sys.argv[5].split(",") if len(sys.argv) > 5 else list(CONFIGS)
    jobs = [(c, g, s, sd, N) for c in cfgs for g in gens for s in scales for sd in range(seeds)]
    t0 = time.time()
    agg = {}
    with Pool(4) as p:
        for cname, g, scale, tot, cmp, ex in p.imap_unordered(work, jobs):
            key = (cname, g, scale)
            A = agg.setdefault(key, [Counter(), Counter(), Counter(), {}])
            A[0].update(tot["main"]); A[1].update(tot["pr"]); A[2].update(cmp)
            for k, e in ex.items():
                if k not in A[3] or e[0] < A[3][k][0]: A[3][k] = e
    for key in sorted(agg):
        A = agg[key]
        print(key, "layouts", A[0]["layouts"], "| main", {k: A[0][k] for k in KEYS}, "| pr", {k: A[1][k] for k in KEYS})
        print("    cmp", dict(sorted(A[2].items())))
        for k, e in A[3].items():
            print("    EX", k, e)
    # per-config totals
    for c in cfgs:
        T = [Counter(), Counter(), Counter()]
        for key, A in agg.items():
            if key[0] == c:
                T[0].update(A[0]); T[1].update(A[1]); T[2].update(A[2])
        print("TOTAL", c, "layouts", T[0]["layouts"], "main", {k: T[0][k] for k in KEYS}, "pr", {k: T[1][k] for k in KEYS})
        print("      ", dict(sorted(T[2].items())))
    print("time", round(time.time() - t0, 1), file=sys.stderr)
