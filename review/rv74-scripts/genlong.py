import random, json, sys
rng = random.Random(int(sys.argv[1])); out = []
for _ in range(int(sys.argv[2])):
    t = rng.choice([0.0, rng.uniform(0, 6)]); segs = []; total = rng.uniform(80, 400)
    while t < total:
        sp = rng.uniform(0.3, 10) if rng.random() < 0.4 else rng.uniform(20, 150)
        e = min(total, t + sp)
        if e - t > 0.05: segs.append((round(t, 3), round(e, 3)))
        r = rng.random(); t = e + (rng.uniform(0.05, 1.5) if r < 0.6 else rng.uniform(1.5, 3.5) if r < 0.85 else rng.uniform(3, 12))
    if rng.random() < 0.5 and segs: total = segs[-1][1] + rng.choice([0, rng.uniform(0, 1), rng.uniform(0, 6)])
    out.append((segs, round(total, 3)))
json.dump(out, open(sys.argv[3], 'w'))
