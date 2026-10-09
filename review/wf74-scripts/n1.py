import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/wf74-scripts")
from load import *
for L in (55.0, 55.1, 59.9, 90.0):
    for k in ("55", "74"):
        p, sg, t = run(MODS[k], [(0, L)], L, target=25.0, mx=30.0, ctx=0.0)
        print("whisper", L, k, show(p.ranges))
# main
import subprocess
