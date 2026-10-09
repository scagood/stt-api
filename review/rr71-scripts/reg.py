from h import *
import fuzz
fuzz.rng = np.random.default_rng(3)
om = 20 * SR
def bad(r, segs):
    return [e for i, (s, e) in enumerate(r[:-1]) if r[i+1][0] == e for a, z in segs if a < e < z and z - a <= om]
for _ in range(3000):
    sp, seconds = fuzz.layout()
    if not sp: continue
    spS = [tuple(int(x * SR) for x in s) for s in sp]; total = int(seconds * SR)
    pn = plan(new, spS, total, BOUNDS["v2"]); po = plan(old, spS, total, BOUNDS["v2"])
    bn, bo = bad(pn.ranges, pn.speech), bad(po.ranges, po.speech)
    if len(bn) > len(bo):
        print("new bad cuts", [x/SR for x in bn], "old bad cuts", [x/SR for x in bo])
        lo = min(bn) - 60 * SR; hi = min(bn) + 30 * SR
        print("speech", [(round(a/SR,2), round(z/SR,2)) for a, z in pn.speech if lo < z and a < hi])
        print("new", [(round(a/SR,2), round(z/SR,2)) for a, z in pn.ranges if lo < z and a < hi])
        print("old", [(round(a/SR,2), round(z/SR,2)) for a, z in po.ranges if lo < z and a < hi])
        break
