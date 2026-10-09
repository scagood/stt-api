import sys, json
root, lay_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, root)
import numpy as np
from parakeet_service import chunker
SR = chunker.TARGET_SR
CFG = {'v2c5': (25, 30, 20, 5), 'v2c0': (25, 30, 20, 0), 'v3c5': (60, 75, 20, 5), 'v3c0': (60, 75, 20, 0)}
lays = json.load(open(lay_path))
res = {}
for name, (tg, mx, mn, cx) in CFG.items():
    rr = []
    for segs, total in lays:
        seg = [(int(round(a * SR)), int(round(b * SR))) for a, b in segs]
        tot = int(round(total * SR))
        chunker._speech_segments = lambda _w, seg=seg: seg
        p = chunker.plan_chunks(np.zeros(tot, dtype=np.float32), target_sec=tg, max_sec=mx, min_sec=mn, context_sec=cx)
        rr.append((p.ranges, p.windows))
    res[name] = rr
json.dump(res, open(out_path, 'w'))
