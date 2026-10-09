import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/rr74-scripts")
from load import *
cases = {
  "lead after cut: (0,33.9)+(35.3,54.6) total 54.6": ([(0, 33.9), (35.3, 54.6)], 54.6),
  "lead after cut, 2-piece next: (0,33.9)+(35.3,74.4) total 80": ([(0, 33.9), (35.3, 74.4)], 80),
  "worse-than-74d1902 seed3#1686": ([(0.58, 8.325), (9.389, 13.111), (14.725, 39.803), (40.236, 60.016)], 60.016),
  "tail margin shrinks: (0,39.9) total 45": ([(0, 39.9)], 45),
}
for name, (segs, total) in cases.items():
    print(name)
    for k, m in MODS.items():
        r, w, sg, t = run(m, segs, total)
        print(f"  {k:12s} {show(r)} {sec(r)} cuts-in-speech={in_speech_cuts(r, sg)}")
