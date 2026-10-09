import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/wf74-scripts")
from load import *
cases = {
  "38.5 s speech at end, silence after (total 45)": ([(0, 38.5)], 45.0),
  "38.5 s speech at end, total 41.5": ([(0, 38.5)], 41.5),
  "38.5 s after earlier speech + long gap": ([(0, 10), (20, 58.5)], 65.0),
  "39.8 + 1 s pause + 5 s phrase": ([(0, 39.8), (40.8, 45.8)], 50.0),
}
for name, (segs, total) in cases.items():
    print(name)
    for k, m in MODS.items():
        plan, sg, t = run(m, segs, total)
        print(f"  {k:4s} {show(plan.ranges)} ranges={[(round(a/SR,2), round(b/SR,2)) for a,b in plan.ranges]} in-speech cuts={in_speech_cuts(plan.ranges, sg)}")
