import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/r76a-scripts")
from load76 import *
D = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/r76a-scripts"
M = {"main": load("main"), "head": load("head")}
for k in ("noguard", "nolead", "neither"): M[k] = load(k, f"{D}/chunker_{k}.py")
M["fixgrid"] = load("fixgrid", "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/c74r-scripts/chunker_fixgrid.py")
cases = [
  ("repro", [(0, 18.8), (21.5, 43.5)], 43.5),
  ("lead-after-cut", [(0, 10), (11.5, 51.45), (54, 93.9)], 97),
]
for t in (60.73, 61, 62, 63, 64, 65, 66, 70):
  cases.append((f"ex1 total {t}", [(2.93, 4.25), (4.58, 60.73)], t))
for name, segs, total in cases:
    for k, mod in M.items():
        r, w, sg, t = run(mod, segs, total)
        print(f"{name:18s} {k:8s} n={len(r)} isc={in_speech_cuts(r, sg)} silent={silent_ranges(r, sg)} {sec(r)}")
