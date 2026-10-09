import sys, collections
from h import *
import fuzz; from fuzz import layout
b = sys.argv[3]; om = int((BOUNDS[b]["max_sec"] - 2 * BOUNDS[b]["context_sec"]) * SR)
shown = collections.Counter()
for _ in range(int(sys.argv[2])):
    sp, seconds = layout()
    if not sp: continue
    spS = [tuple(int(x * SR) for x in s) for s in sp]; total = int(seconds * SR)
    p = plan(new, spS, total, BOUNDS[b])
    if not p.speech: continue
    r = p.ranges; segs = p.speech
    for i, (s, e) in enumerate(r):
        if not any(a < e and s < z for a, z in segs) and shown["silent"] < 4:
            shown["silent"] += 1
            near = [(round(a/SR,2), round(z/SR,2)) for a, z in segs if a < e + 30*SR and z > s - 30*SR]
            print("SILENT", (s/SR, e/SR), "idx", i, "/", len(r), "total", total/SR, "nearby speech", near, "ranges", [(a/SR, z/SR) for a, z in r[max(0,i-2):i+2]])
    for i, (s, e) in enumerate(r[:-1]):
        c = e
        if r[i+1][0] != c: continue
        for a, z in segs:
            if a < c < z and z - a <= om and shown["fit"] < 4:
                shown["fit"] += 1
                near = [(round(x/SR,2), round(y/SR,2)) for x, y in segs if x < c + 25*SR and y > c - 25*SR]
                print("CUT-IN-FITTING", c/SR, "in", (a/SR, z/SR), "nearby", near, "ranges", [(round(x/SR,2), round(y/SR,2)) for x, y in r[max(0,i-2):i+3]])
