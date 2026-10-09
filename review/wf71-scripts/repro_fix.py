from loadmods import fx, pr, mn, SR
import numpy as np
def run(m, sp, tot):
    m._speech_segments = lambda _w: [(int(round(a*SR)), int(round(b*SR))) for a,b in sp]
    return [(a/SR,b/SR) for a,b in m.plan_chunks(np.zeros(int(tot*SR),dtype=np.float32), target_sec=25.0, max_sec=30.0, min_sec=20.0, context_sec=5.0).ranges]
for sp, tot in (([(3,13),(14,33.8)],40), ([(3,19.8),(20.6,39.5)],45), ([(2,21.5),(22.5,30)],40)):
    print(sp, "fix:", run(fx, sp, tot))
