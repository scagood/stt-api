import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, ".")
import random, bisect
from collections import defaultdict
from lib import *
VERS["fix1"] = fix1; VERS["fix2"] = fix2; VERS["fix3"] = fix3
seed = int(sys.argv[2]); N = int(sys.argv[3])
rng = random.Random(seed)
OWN = {"v2": 20*SR, "v3": 65*SR, "wh": 30*SR}
def gen():
    seconds = rng.uniform(60, 600)
    at = rng.uniform(0, 5); sp = []
    while True:
        r = rng.random()
        L = rng.uniform(2,6) if r < .8 else rng.uniform(6,22) if r < .95 else rng.uniform(22,70)
        if at + L > seconds - 1: break
        sp.append(at(at, at+L) if False else (int(at*SR), int((at+L)*SR)))
        at += L + rng.uniform(0.25, 12)
    return sp, int(seconds*SR)
files = [gen() for _ in range(N)]
files = [f for f in files if f[0]]
res = {}
for model in ["v2", "v3", "wh"]:
    own = OWN[model]
    for name, mod in VERS.items():
        c = defaultdict(int)
        outs = []
        for sp, total in files:
            r = plan(mod, model, speech=sp, total=total).ranges
            outs.append(r)
            starts = [s for s, e in sp]
            for (a, b), (a2, b2) in zip(r, r[1:]):
                if b != a2: continue
                i = bisect.bisect_right(starts, b) - 1
                inside = i >= 0 and sp[i][0] < b < sp[i][1]
                if inside:
                    c["inspeech"] += 1
                    if sp[i][1] - sp[i][0] <= own: c["infit"] += 1
                j = bisect.bisect_left(starts, b)
                if j < len(sp) and 0 <= sp[j][0] - b <= 0.05*SR and not inside: c["onset"] += 1
            for a, b in r:
                if not any(s < b and e > a for s, e in sp): c["silent"] += 1
                if b - a > own: c["over"] += 1
            if r:
                m = r[-1][1] - sp[-1][1]
                if m == 0: c["margin0"] += 1
                if m < 0.5*SR: c["margin<0.5"] += 1
            c["ranges"] += len(r)
            c["frag<2s"] += sum(1 for a, b in r if b - a < 2*SR)
        res[(model, name)] = (dict(c), outs)
        print(model, name, dict(sorted(c.items())))
# per-file regressions new vs old (in-fit cuts)
