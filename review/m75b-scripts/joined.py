import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/m75b-scripts")
from lib import *
def train(t0, t1, period, dur, lvl):
    out = []; t = t0
    while t < t1:
        out.append((t, dur, lvl)); t += period
    return out
cases = [
  ("typing 5/s, 60ms keys -40, 45-55", train(45, 55, 0.2, 0.06, -40)),
  ("typing 8/s, 40ms keys -40, 45-55", train(45, 55, 0.125, 0.04, -40)),
  ("clock 1/s 20ms -35, 40-60", train(40.5, 60, 1.0, 0.02, -35)),
  ("clock 4/s 20ms -35 (fast ticker)", train(40.5, 60, 0.25, 0.02, -35)),
  ("footsteps 150ms every 0.5s -40, 44-50", train(44, 50, 0.5, 0.15, -40)),
  ("footsteps 150ms every 0.6s -40, 44-50", train(44, 50, 0.6, 0.15, -40)),
  ("rain-ish 10ms clicks every 30ms -45, 45-50", train(45, 50, 0.03, 0.01, -45)),
  ("breaths 300ms every 0.65s (panting) -42, 45-50", train(45, 50, 0.65, 0.3, -42)),
  ("breaths 300ms every 0.75s -42, 45-50", train(45, 50, 0.75, 0.3, -42)),
]
for label, s in cases:
    report(label, build(20, s))
