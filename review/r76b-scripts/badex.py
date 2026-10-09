import random
from collections import Counter
from h import *
from fz import gen
cfg = CONFIGS["v2c5"]; rng = random.Random(5); c = Counter(); shown = 0
for i in range(20000):
    segs, total = gen(rng, cfg, 1.0, "mix")
    r, w, sp, t = run(MODS["main"], segs, total, cfg)
    x = metrics(r, w, sp, t, cfg)
    for b in x["badl"]: c[b] += 1
    if x["bad"] and shown < 2:
        shown += 1; print(segs, total, x["badl"]); print("  r", sec(r)); print("  w", sec(w)); print("  wl", lens(w))
print(c)
