import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/c74r-scripts")
from load import *
cases = {
  "(3,10)+(10.5,41.5) total 45": ([(3, 10), (10.5, 41.5)], 45),
  "fuzz: (0.403,17.654)+(18.17,18.804)+(21.053,40.008)+(46.427,54.28) total 54.28": ([(0.403, 17.654), (18.17, 18.804), (21.053, 40.008), (46.427, 54.28)], 54.28),
  "(3,41.5) total 45": ([(3, 41.5)], 45),
}
for name, (segs, total) in cases.items():
    print(name)
    for k, m in MODS.items():
        r, w, sg, t = run(m, segs, total)
        print(f"  {k:9s} {show(r)} {sec(r)} isc={in_speech_cuts(r, sg)} silent={silent_ranges(r, sg)}")
