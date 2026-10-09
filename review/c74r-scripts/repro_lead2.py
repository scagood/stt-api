import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/c74r-scripts")
from load import *
cases = {
  "lead, two phrases (3,10)+(10.5,41.5) total 45": ([(3, 10), (10.5, 41.5)], 45),
  "lead, two phrases (3,25)+(25.5,41.5) total 45": ([(3, 25), (25.5, 41.5)], 45),
  "lead, (3,8)+(8.5,13)+(13.5,41.5) total 45": ([(3, 8), (8.5, 13), (13.5, 41.5)], 45),
  "lead 2.5 (2.5,10)+(10.5,41.5) total 41.5": ([(2.5, 10), (10.5, 41.5)], 41.5),
}
for name, (segs, total) in cases.items():
    print(name)
    for k, m in MODS.items():
        r, w, sg, t = run(m, segs, total)
        print(f"  {k:9s} {show(r)} {sec(r)} isc={in_speech_cuts(r, sg)} silent={silent_ranges(r, sg)}")
