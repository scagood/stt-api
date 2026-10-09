import sys, ast
from h import *
cname = sys.argv[1]; segs = ast.literal_eval(sys.argv[2]); total = float(sys.argv[3])
cfg = CONFIGS[cname]
for k, mod in MODS.items():
    r, w, sp, t = run(mod, segs, total, cfg)
    x = metrics(r, w, sp, t, cfg)
    print(k, "ranges", sec(r), lens(r))
    print("   windows", sec(w), lens(w))
    print("   ", {a: x[a] for a in ("n", "isc", "silent", "short", "lost", "wedge")}, x["badl"])
    for (a, b), (wa, wb) in zip(r, w):
        flags = []
        if wa != a and inside(wa, sp): flags.append("start@%.2f in speech" % (wa / SR))
        if wb != b and inside(wb, sp): flags.append("end@%.2f in speech" % (wb / SR))
        if flags: print("     range", sec([(a, b)]), flags)
