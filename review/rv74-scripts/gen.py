import random, json, sys
def layouts(seed, n):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        total = rng.uniform(31, 200)
        t = rng.choice([0.0, 0.0, rng.uniform(0, 1.5), rng.uniform(0, 6)])
        segs = []
        while t < total:
            r = rng.random()
            sp = rng.uniform(0.3, 8) if r < 0.5 else rng.uniform(5, 45)
            e = min(total, t + sp)
            if e - t > 0.05:
                segs.append((round(t, 3), round(e, 3)))
            r = rng.random()
            pa = rng.uniform(0.05, 1.5) if r < 0.6 else (rng.uniform(1.5, 3.5) if r < 0.85 else rng.uniform(3, 12))
            t = e + pa
        if rng.random() < 0.5 and segs:
            # trailing silence
            total = segs[-1][1] + rng.choice([0, rng.uniform(0, 1), rng.uniform(0, 6)])
        out.append((segs, round(total, 3)))
    return out
if __name__ == '__main__':
    json.dump(layouts(int(sys.argv[1]), int(sys.argv[2])), open(sys.argv[3], 'w'))
