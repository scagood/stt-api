from loadmods import load, pr, mn, SR, S
import numpy as np
f3 = load("f3_pkg", S + "/wf71-scripts/fixF3")
def run(m, sp, tot, **kw):
    b = dict(target_sec=25.0, max_sec=30.0, min_sec=20.0, context_sec=5.0); b.update(kw)
    m._speech_segments = lambda _w: [(int(round(a*SR)), int(round(b_*SR))) for a,b_ in sp]
    return [(a/SR,b_/SR) for a,b_ in m.plan_chunks(np.zeros(int(tot*SR),dtype=np.float32), **b).ranges]
for sp, tot in (([(3,40),(41,50)],60), ([(3,40)],60), ([(3,19.8),(20.6,39.5)],45)):
    print(sp, "\n  main:", run(mn, sp, tot), "\n  pr:  ", run(pr, sp, tot), "\n  f3:  ", run(f3, sp, tot))
# v3
for sp, tot in (([(3,130),(131,150)],200), ([(3,130)],200)):
    kw = dict(target_sec=60.0, max_sec=75.0, min_sec=20.0, context_sec=5.0)
    print("v3", sp, "\n  pr:  ", run(pr, sp, tot, **kw), "\n  f3:  ", run(f3, sp, tot, **kw))
