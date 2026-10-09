"""Stage 3 (real model): redo windows whose input ends just after a word that
follows a pause (or starts just before one), the edges where the redo's
timing is furthest from the piece's. Appends JSON lines like stage 1."""
import json, random, sys
import realmodel as rm
SR = 16000
CLIP = 30.0
src, out, per_clip = sys.argv[1], open(sys.argv[2], "a"), int(sys.argv[3])
rng = random.Random(3)
cache = {}
for line in open(src):
    rec = json.loads(line)
    toks, ts = rec["ptoks"], rec["pts"]
    starts = [t for tok, t in zip(toks, ts) if tok.startswith((" ", "▁"))]
    # words after a pause of >= 0.5 s between starts
    after_pause = [starts[k] for k in range(1, len(starts)) if starts[k] - starts[k - 1] >= 0.5 and 11.0 <= starts[k] <= CLIP - 0.5]
    before_pause = [starts[k] for k in range(len(starts) - 1) if starts[k + 1] - starts[k] >= 0.5 and 0.5 <= starts[k] <= CLIP - 11.0]
    if rec["name"] not in cache:
        cache[rec["name"]] = rm.audio(rec["name"])
    wav = cache[rec["name"]]
    clip = wav[int(rec["off"] * SR): int((rec["off"] + CLIP) * SR)]
    holes = []
    for t in rng.sample(after_pause, min(per_clip, len(after_pause))):
        b = min(CLIP, t + rng.choice([0.08, 0.16, 0.24]))
        a = b - rng.uniform(8, 11)
        r = rm.recognize(clip[int(round(a * SR)): int(round(b * SR))])
        holes.append(dict(kind="end", s=a, e=a, low=a, high=b, a=a, b=b, rtoks=list(r.tokens), rts=[float(x) for x in r.timestamps]))
    for t in rng.sample(before_pause, min(per_clip, len(before_pause))):
        a = max(0.0, t + rng.choice([0.0, 0.08, 0.16]))
        b = a + rng.uniform(8, 11)
        r = rm.recognize(clip[int(round(a * SR)): int(round(b * SR))])
        holes.append(dict(kind="start", s=a, e=a, low=a, high=b, a=a, b=b, rtoks=list(r.tokens), rts=[float(x) for x in r.timestamps]))
    out.write(json.dumps(dict(name=rec["name"], off=rec["off"], ptoks=toks, pts=ts, holes=holes)) + "\n")
    out.flush()
    print(rec["name"], round(rec["off"], 1), len(holes), flush=True)
