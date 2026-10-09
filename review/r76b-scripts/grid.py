import sys, itertools
from collections import Counter
from multiprocessing import Pool
from h import *
KEYS = ("n", "isc", "silent", "short", "lost", "bad")
cname = sys.argv[1]; cfg = CONFIGS[cname]; own = own_of(cfg); u = own / 20.0
F = [0, 1.5, 3, 4.5]; A = [x * u for x in range(1, 21)]; G1 = [0.25, 1, 2, 2.75, 2.95]
L2 = [own * 0.75 + 0.5 * i * u for i in range(int((own * 3.5 - own * 0.75) / (0.5 * u)))]
G2 = [0.5, 2, 2.9, 3.5]; L3 = [1 * u, 5 * u, 12 * u, own * 0.95, own * 1.1, own * 1.5, own * 2.2]; TAIL = [0, 1.2, 4]
def work(f):
    C = Counter(); ex = {}
    for a, g1, l2, g2, l3, tail in itertools.product(A, G1, L2, G2, L3, TAIL):
        segs = [(f, f + a), (f + a + g1, f + a + g1 + l2)]
        s3 = segs[-1][1] + g2; segs.append((s3, s3 + l3))
        segs = [(round(x, 4), round(y, 4)) for x, y in segs]; total = round(segs[-1][1] + tail, 4)
        m = {}
        for k, mod in MODS.items():
            r, w, sp, t = run(mod, segs, total, cfg); m[k] = metrics(r, w, sp, t, cfg); m[k]["r"] = r
        C["layouts"] += 1
        if m["pr"]["r"] != m["main"]["r"]: C["differ"] += 1
        for key in KEYS + ("wedge",):
            if m["pr"][key] > m["main"][key]:
                C[key + "_worse"] += 1
                if key not in ex: ex[key] = (segs, total, sec(m["main"]["r"]), sec(m["pr"]["r"]))
            elif m["pr"][key] < m["main"][key]: C[key + "_better"] += 1
    return C, ex
if __name__ == "__main__":
    T = Counter(); EX = {}
    with Pool(4) as p:
        for C, ex in p.imap_unordered(work, F):
            T.update(C); EX.update({k: v for k, v in ex.items() if k not in EX})
    print(cname, dict(sorted(T.items())))
    for k, v in EX.items():
        if k != "wedge": print("  EX", k, v)
