import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/rr74-scripts")
from load import *
cases = {
  "F1a speech (0,38.5) total 41.5": ([(0, 38.5)], 41.5),
  "F1b speech (0,38.5) total 45": ([(0, 38.5)], 45.0),
  "F1c (0,39.8)+(40.8,45.8) total 50": ([(0, 39.8), (40.8, 45.8)], 50.0),
  "38.5 s after earlier speech + long gap": ([(0, 10), (20, 58.5)], 65.0),
  "mirror: speech (3,41.5) total 45": ([(3, 41.5)], 45.0),
}
for name, (segs, total) in cases.items():
    print(name)
    for k, m in MODS.items():
        r, w, sg, t = run(m, segs, total)
        print(f"  {k:12s} {show(r)} {sec(r)} cuts-in-speech={in_speech_cuts(r, sg)}")
