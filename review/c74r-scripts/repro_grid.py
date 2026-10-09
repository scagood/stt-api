import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/c74r-scripts")
from load import *
cases = {
  "grid (0,18.8)+(21.5,43.5) total 43.5": ([(0, 18.8), (21.5, 43.5)], 43.5),
  "grid (0,18.8)+(21.5,43.5) total 48": ([(0, 18.8), (21.5, 43.5)], 48),
  "grid fuzz range (144.371,163.195)+(165.862,187.922) then long gap": ([(144.371, 163.195), (165.862, 187.922), (193.793, 200)], 200),
}
for name, (segs, total) in cases.items():
    print(name)
    for k, m in MODS.items():
        r, w, sg, t = run(m, segs, total)
        print(f"  {k:9s} {show(r)} {sec(r)} isc={in_speech_cuts(r, sg)} silent={silent_ranges(r, sg)}")
