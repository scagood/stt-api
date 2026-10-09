import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/r76a-scripts")
from load76 import *
M = {k: load(k) for k in ("main", "head")}
cases = [
  ("repro", [(0, 18.8), (21.5, 43.5)], 43.5),
  ("lead-after-cut", [(0, 10), (11.5, 51.45), (54, 93.9)], 97),
  ("cited ex1 total 61", [(2.93, 4.25), (4.58, 60.73)], 61),
  ("cited ex1 total 65", [(2.93, 4.25), (4.58, 60.73)], 65),
  ("cited ex1 total 60.73", [(2.93, 4.25), (4.58, 60.73)], 60.73),
]
for name, segs, total in cases:
    for k, mod in M.items():
        r, w, sg, t = run(mod, segs, total)
        print(f"{name:22s} {k:5s} n={len(r)} isc={in_speech_cuts(r, sg)} silent={silent_ranges(r, sg)} lost={speech_lost(r, sg)} {sec(r)}")
