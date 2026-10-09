import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/c74r-scripts")
import random
from load import *
from fuzz import layout, CONFIGS
cfg = CONFIGS["v2 ctx5"]; shown = 0
for seed in (1, 2):
    rng = random.Random(seed * 100)
    for i in range(3000):
        segs, total = layout(rng)
        if total <= cfg["mx"]: continue
        rn = run(MODS["c73d2b9"], segs, total, **cfg); rf = run(MODS["fixgrid"], segs, total, **cfg)
        if len(rf[0]) > len(rn[0]) and shown < 2:
            shown += 1; sg = rn[2]
            print(segs, total); print("  head", sec(rn[0]), in_speech_cuts(rn[0], sg)); print("  fix ", sec(rf[0]), in_speech_cuts(rf[0], sg))
