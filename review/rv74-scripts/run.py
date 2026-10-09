import sys, json
root = sys.argv[1]
sys.path.insert(0, root)
import numpy as np
from parakeet_service import chunker
SR = chunker.TARGET_SR
def at(*xs): return tuple(int(round(x*SR)) for x in xs)
def plan(speech, total, target=25.0, mx=30.0, mn=20.0, ctx=5.0):
    chunker._speech_segments = lambda _w: [at(*s) for s in speech]
    p = chunker.plan_chunks(np.zeros(at(total)[0], dtype=np.float32), target_sec=target, max_sec=mx, min_sec=mn, context_sec=ctx)
    return [(round(a/SR,3), round(b/SR,3)) for a,b in p.ranges], [(round(a/SR,3), round(b/SR,3)) for a,b in p.windows]
cases = json.loads(sys.argv[2])
for c in cases:
    r, w = plan(*c)
    print(c, '->', r)
