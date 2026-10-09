from loadmods import load, pr, mn, SR, S
import numpy as np
f2 = load("f2_pkg", S + "/wf71-scripts/fixF2")
def run(m, sp, tot):
    m._speech_segments = lambda _w: [(int(round(a*SR)), int(round(b*SR))) for a,b in sp]
    return [(a/SR,b/SR) for a,b in m.plan_chunks(np.zeros(int(tot*SR),dtype=np.float32), target_sec=25.0, max_sec=30.0, min_sec=20.0, context_sec=5.0).ranges]
for sp, tot in (([(2,21.5),(22.5,30)],40), ([(2,21.9),(23.5,30)],40), ([(2,21.5),(26,40)],50), ([(2,21.5)],40)):
    print(sp, "\n  main:", run(mn, sp, tot), "\n  pr:  ", run(pr, sp, tot), "\n  f2:  ", run(f2, sp, tot))
