import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/r76a-scripts")
import itertools, random
from collections import Counter
from multiprocessing import Pool
from fuzz76 import CONFIGS, EXTRA, metrics, init
import fuzz76
from load76 import *
TG = 48000
def vals(cfg):
    own = max(1, int(cfg["mx"] * SR) - 2 * int(cfg["ctx"] * SR))
    L = sorted({x for x in [own // 10, own // 2, own - TG - 1, own - TG, own - TG + 1, own - 1, own, own + 1, own + TG,
                            2 * own - TG - 1, 2 * own - TG, 2 * own - TG + 1, 2 * own - 1, 2 * own, 2 * own + 1] if x > 0})
    P = [6400, 16000, TG // 2, TG - 1]
    lead = [0, 1, TG - 1, TG, 2 * TG]
    tail = [0, 1, TG, 3 * TG]
    return own, L, P, lead, tail
def layouts(cfg, nseg):
    own, L, P, lead, tail = vals(cfg)
    for ls in itertools.product(L, repeat=nseg):
        for ps in itertools.product(P, repeat=nseg - 1):
            for ld in lead:
                for tl in tail:
                    t = ld; segs = []
                    for i, l in enumerate(ls):
                        segs.append((t, t + l)); t += l + (ps[i] if i < nseg - 1 else 0)
                    yield segs, t + tl
def job(args):
    cname, nseg, part, parts = args
    cfg = {**CONFIGS, **EXTRA}[cname]
    M = fuzz76.MODS
    c = Counter(); ex = {}
    for idx, (segs, total) in enumerate(layouts(cfg, nseg)):
        if idx % parts != part: continue
        if total <= int(cfg["mx"] * SR): continue
        m = {}
        for k, mod in M.items():
            r, w, sg, t = run(mod, segs, total, samples=True, **cfg)
            m[k] = metrics(r, w, sg, t, cfg); m[k]["r"] = r
        c["layouts"] += 1
        a, b = m["head"], m["main"]
        if a["r"] != b["r"]: c["changed"] += 1
        for key in ("n", "isc", "silent", "short", "lost"):
            if a[key] > b[key]: c[key + "_worse"] += 1; ex.setdefault(key, (segs, total, b["r"], a["r"]))
            elif a[key] < b[key]: c[key + "_better"] += 1
        if a["bad"]: c["bad"] += 1; ex.setdefault("bad", (segs, total, a["bad"]))
    return cname, nseg, c, ex
if __name__ == "__main__":
    jobs = [(cn, ns, p, 8) for cn in ("v2 ctx5", "v2 ctx0", "v3 ctx5") for ns in (2, 3) for p in range(8)]
    agg = {}
    with Pool(4, initializer=init) as pool:
        for cname, nseg, c, ex in pool.imap_unordered(job, jobs):
            A = agg.setdefault((cname, nseg), [Counter(), {}]); A[0].update(c)
            for k, v in ex.items(): A[1].setdefault(k, v)
    for k in sorted(agg):
        print(k, dict(sorted(agg[k][0].items())))
        for kk, v in agg[k][1].items(): print("   ex", kk, v)
