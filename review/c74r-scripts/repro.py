import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/c74r-scripts")
from load import *
cases = {
  "F1a (0,38.5) total 41.5": ([(0, 38.5)], 41.5),
  "F1b (0,38.5) total 45": ([(0, 38.5)], 45.0),
  "F1c (0,39.8)+(40.8,45.8) total 50": ([(0, 39.8), (40.8, 45.8)], 50.0),
  "(0,10)+(20,58.5) total 65": ([(0, 10), (20, 58.5)], 65.0),
  "lead (3,41.5) total 45": ([(3, 41.5)], 45.0),
  "lead-after-cut (0,33.9)+(35.3,54.8) total 60": ([(0, 33.9), (35.3, 54.8)], 60.0),
  "lead-after-cut (0,33.9)+(35.3,54.8) total 54.8": ([(0, 33.9), (35.3, 54.8)], 54.8),
  "(0,33.9)+(35.3,74.8) total 80": ([(0, 33.9), (35.3, 74.8)], 80),
  "tail (0,39.9) total 45": ([(0, 39.9)], 45),
  "no-pause 40.1": ([(0, 40.1)], 40.1),
}
for name, (segs, total) in cases.items():
    print(name)
    for k, m in MODS.items():
        r, w, sg, t = run(m, segs, total)
        print(f"  {k:9s} {show(r)} {sec(r)} isc={in_speech_cuts(r, sg)} silent={silent_ranges(r, sg)} lost={speech_lost(r, sg)}")
