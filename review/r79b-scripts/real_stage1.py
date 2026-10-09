"""Stage 1 (real Parakeet v3 int8): decode 30 s clips, cut a synthetic hole in
each decode, decode the PR's redo window for that hole. Writes JSON lines."""
import json, random, sys, time
import numpy as np
import realmodel as rm
SR = 16000
out = open(sys.argv[1], "a")
names = sys.argv[2].split(",")
nclips = int(sys.argv[3])
seed = int(sys.argv[4])
rng = random.Random(seed)
CLIP = 30.0
for name in names:
    wav = rm.audio(name)
    total = wav.size / SR
    offsets = sorted(rng.uniform(60, total - CLIP - 1) for _ in range(nclips))
    for off in offsets:
        clip = wav[int(off * SR): int((off + CLIP) * SR)]
        t0 = time.time()
        p = rm.recognize(clip)
        ptoks, pts = list(p.tokens), [float(x) for x in p.timestamps]
        holes = []
        for _ in range(4):
            s = rng.uniform(3.0, CLIP - 10.0)
            L = rng.uniform(3.6, 8.0)
            e = s + L
            # snap to word starts (a token opening a word: leading space or marker)
            starts = [t for tok, t in zip(ptoks, pts) if tok.startswith((" ", "\u2581"))]
            s = next((t for t in starts if t >= s), CLIP)
            e = next((t for t in starts if t >= e), CLIP)
            keep = [t for t in pts if not (s <= t < e)]
            before = [t for t in keep if t < s]
            after = [t for t in keep if t >= e]
            low = (before[-1] + 0.32) if before else 0.0
            high = after[0] if after else CLIP
            a, b = max(0.0, low - 2.0), min(CLIP, high + 2.0)
            if (a, b) == (0.0, CLIP):
                a, b = low, high
            r = rm.recognize(clip[int(round(a * SR)): int(round(b * SR))])
            holes.append(dict(s=s, e=e, low=low, high=high, a=a, b=b, rtoks=list(r.tokens), rts=[float(x) for x in r.timestamps]))
        out.write(json.dumps(dict(name=name, off=off, ptoks=ptoks, pts=pts, holes=holes)) + "\n")
        out.flush()
        print(name, round(off, 1), "done in", round(time.time() - t0, 1), flush=True)
