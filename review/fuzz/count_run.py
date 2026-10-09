import sys, json, numpy as np
sys.path.insert(0, sys.argv[1])
from parakeet_service import chunker
out = []
for sp, total in json.load(open(sys.argv[2])):
    sp = [tuple(x) for x in sp]
    chunker._speech_segments = lambda _w: sp
    r = []
    for t, mx, ctx in [(25, 30, 5), (60, 75, 5), (25, 30, 0)]:
        r.append(len(chunker.auto_chunk(np.broadcast_to(np.float32(0), (total,)), target_sec=t, max_sec=mx, min_sec=20, context_sec=ctx)))
    out.append(r)
json.dump(out, open(sys.argv[3], 'w'))
