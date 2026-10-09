import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/rr74-scripts")
from load import *
cases = {
  "(0,33.9)+(35.3,54.8) total 54.8": ([(0, 33.9), (35.3, 54.8)], 54.8),
  "(0,33.9)+(35.3,74.8) total 80": ([(0, 33.9), (35.3, 74.8)], 80),
  "(0,33.9)+(35.3,54.8)+(60,70) total 70": ([(0, 33.9), (35.3, 54.8), (60, 70)], 70),
}
for name, (segs, total) in cases.items():
    print(name)
    for k, m in MODS.items():
        r, w, sg, t = run(m, segs, total)
        print(f"  {k:12s} {show(r)} {sec(r)} cuts-in-speech={in_speech_cuts(r, sg)}")
