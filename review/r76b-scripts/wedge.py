import random, sys
from collections import Counter
from h import *
from fz import gen
def edges(r, w, segs):
    ws = sum(1 for (a, b), (wa, wb) in zip(r, w) if wa != a and inside(wa, segs))
    we = sum(1 for (a, b), (wa, wb) in zip(r, w) if wb != b and inside(wb, segs))
    return ws, we
cname = sys.argv[1]; g = sys.argv[2]; scale = float(sys.argv[3]); N = int(sys.argv[4])
cfg = CONFIGS[cname]; rng = random.Random(77)
joint = Counter(); E = Counter(); ex = {}
for i in range(N):
    segs, total = gen(rng, cfg, scale, g)
    if total <= cfg["mx"]: continue
    out = {}
    for k, mod in MODS.items():
        r, w, sp, t = run(mod, segs, total, cfg)
        isc = sum(1 for (a, b), (c, d) in zip(r, r[1:]) if b == c and inside(b, sp))
        out[k] = (r, w, sp, isc) + edges(r, w, sp)
    m, p = out["main"], out["pr"]
    if m[0] == p[0]: continue
    d = (p[3] - m[3], (p[4] + p[5]) - (m[4] + m[5]))
    joint[d] += 1
    E["ws main"] += m[4]; E["ws pr"] += p[4]; E["we main"] += m[5]; E["we pr"] += p[5]; E["isc main"] += m[3]; E["isc pr"] += p[3]
    E["we worse"] += p[5] > m[5]; E["we better"] += p[5] < m[5]
    E["ws worse"] += p[4] > m[4]; E["ws better"] += p[4] < m[4]
    if d[0] >= 0 and d[1] > 0:
        key = d
        if key not in ex or len(segs) < len(ex[key][0]): ex[key] = (segs, total)
print(cname, g, scale, "differing layouts", sum(joint.values()))
print("  joint (d_isc, d_wedge):", sorted(joint.items(), key=lambda x: -x[1])[:12])
print("  ", dict(E))
for k, v in ex.items(): print("  EX", k, v)
