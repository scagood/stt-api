import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/m75b-scripts")
from lib import *
# 1. quiet "Yes." 600 ms at -40 at 41.0, then breaths 300 ms -42 every 2.5 s through a 20 s pause
P = 20
sounds = [(41.0, 0.6, -40)] + [(41.0 + 2.5 * k, 0.3, -42) for k in range(1, 8)]
report("yes600 + breaths/2.5s in 20s pause", build(P, sounds), show_pauses=True, window=(40, 60))
sounds = [(41.0 + 2.5 * k, 0.3, -42) for k in range(1, 8)]
report("breaths/2.5s only in 20s pause", build(P, sounds), show_pauses=True, window=(40, 60))
# 2. cough 0.6 s mid pause with breaths around
sounds = [(50.0, 0.6, -38)] + [(50.0 + d, 0.3, -42) for d in (-8, -5.5, -2.8, -1.2, 1.5, 2.9, 5.5, 8)]
report("cough600 mid 20s pause + breaths around", build(P, sounds), show_pauses=True, window=(40, 60))
sounds = [(50.0, 0.6, -38)]
report("cough600 alone", build(P, sounds))
