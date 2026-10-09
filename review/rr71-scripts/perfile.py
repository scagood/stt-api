from h import *
import fuzz
fuzz.rng = np.random.default_rng(3)
def incuts(r, segs, om, fit_only):
    n = 0
    for i, (s, e) in enumerate(r[:-1]):
        if r[i+1][0] != e: continue
        n += sum(1 for a, z in segs if a < e < z and (not fit_only or z - a <= om))
    return n
for b in ("v2", "v3", "wh"):
    om = int((BOUNDS[b]["max_sec"] - 2 * BOUNDS[b]["context_sec"]) * SR)
    worse = worse_fit = better = 0; ex = None
    for _ in range(3000):
        sp, seconds = fuzz.layout()
        if not sp: continue
        spS = [tuple(int(x * SR) for x in s) for s in sp]; total = int(seconds * SR)
        pn = plan(new, spS, total, BOUNDS[b]); po = plan(old, spS, total, BOUNDS[b])
        if not pn.speech: continue
        a, c = incuts(pn.ranges, pn.speech, om, False), incuts(po.ranges, po.speech, om, False)
        af, cf = incuts(pn.ranges, pn.speech, om, True), incuts(po.ranges, po.speech, om, True)
        worse += a > c; better += a < c; worse_fit += af > cf
        if af > cf and ex is None: ex = (sp, seconds)
    print(b, "files new>old in-speech cuts:", worse, "new<old:", better, "new>old in fitting phrases:", worse_fit)
    if ex:
        sp, seconds = ex
        # find the minimal local window around the difference
        print("  example", [(round(x,2), round(y,2)) for x, y in sp][:12], round(seconds,1))
