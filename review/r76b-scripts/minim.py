import sys, ast
from h import *
cname, key = sys.argv[1], sys.argv[2]; segs = ast.literal_eval(sys.argv[3]); total = float(sys.argv[4])
cfg = CONFIGS[cname]
def worse(segs, total):
    if not segs or total <= cfg["mx"] or any(b <= a for a, b in segs) or total < segs[-1][1]: return False
    m = {}
    for k, mod in MODS.items():
        r, w, sp, t = run(mod, segs, total, cfg); m[k] = metrics(r, w, sp, t, cfg)
    return m["pr"][key] > m["main"][key]
assert worse(segs, total)
changed = True
while changed:
    changed = False
    # drop segments
    for i in range(len(segs)):
        s2 = segs[:i] + segs[i + 1:]
        for t2 in (total, s2[-1][1] if s2 else 0):
            if s2 and worse(s2, t2): segs, total, changed = s2, t2, True; break
        if changed: break
    if changed: continue
    # shift everything earlier
    for d in (segs[0][0], 1.0, 0.1):
        if d > 0 and segs[0][0] - d >= 0:
            s2 = [(round(a - d, 3), round(b - d, 3)) for a, b in segs]
            if worse(s2, round(total - d, 3)): segs, total, changed = s2, round(total - d, 3), True; break
    if changed: continue
    # trim total
    for t2 in (segs[-1][1], round(total - 1, 3), round(total - 0.1, 3)):
        if t2 < total and worse(segs, t2): total, changed = t2, True; break
    if changed: continue
    # round values
    for q in (1.0, 0.5, 0.1, 0.05):
        for i in range(len(segs)):
            for j in (0, 1):
                v = segs[i][j]; r = round(round(v / q) * q, 3)
                if r != v:
                    s2 = list(segs); s2[i] = (r, s2[i][1]) if j == 0 else (s2[i][0], r)
                    t2 = total if not (i == len(segs) - 1 and j == 1 and total == v) else r
                    if worse(s2, t2): segs, total, changed = s2, t2, True
        r = round(round(total / q) * q, 3)
        if r != total and worse(segs, r): total, changed = r, True
        if changed: break
print(segs, total)
for k, mod in MODS.items():
    r, w, sp, t = run(mod, segs, total, cfg); x = metrics(r, w, sp, t, cfg)
    print(" ", k, sec(r), lens(r), {a: x[a] for a in ("n", "isc", "silent", "short", "wedge")})
