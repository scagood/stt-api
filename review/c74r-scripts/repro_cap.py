import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/c74r-scripts")
from load import *
cases = {
  "cap binds (0,40)+(42.9,81.5) total 85": ([(0, 40), (42.9, 81.5)], 85),
  "cap binds (0,40)+(42.9,81.5) total 81.5": ([(0, 40), (42.9, 81.5)], 81.5),
  "(0,40)+(41,79) total 85": ([(0, 40), (41, 79)], 85),
  "v3 100s no pause": ([(0, 100)], 100),
}
for name, (segs, total) in cases.items():
    print(name)
    for k, m in MODS.items():
        cfg = dict(target=60, mx=75, ctx=5) if "v3" in name else {}
        r, w, sg, t = run(m, segs, total, **cfg)
        print(f"  {k:9s} {show(r)} {sec(r)} isc={in_speech_cuts(r, sg)} silent={silent_ranges(r, sg)}")
