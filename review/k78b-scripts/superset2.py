"""Adversarial superset check: non-stationary floor (piecewise levels), long pauses, quiet phrases,
straddling trains placed just inside a phrase's 3 s reach. usage: superset2.py N SEED MINSIL"""
import sys
import numpy as np
import lib
N, SEED, MINSIL = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
lib.set_minsil(MINSIL)
SR = lib.SR
rng = np.random.default_rng(SEED)
bad = {"split": [], "retime": []}
worse_plan = []
for case in range(N):
    P = rng.uniform(6, 40)
    n = int(P * SR)
    # piecewise floor
    pause = np.zeros(n)
    cuts = np.sort(rng.uniform(0, P, rng.integers(0, 4)))
    edges = [0.0, *cuts, P]
    for a, b in zip(edges, edges[1:]):
        lv = rng.uniform(-64, -46)
        i, j = int(a * SR), int(b * SR)
        pause[i:j] = rng.standard_normal(j - i) * 10 ** (lv / 20)
    base = -55
    speech = []
    t = rng.uniform(0, 2)
    while t < P - 1:
        r = rng.random()
        if r < 0.3:  # quiet phrase
            d = rng.uniform(0.5, 1.5); lv = rng.uniform(-50, -36); u = t
            while u < t + d:
                s = rng.uniform(0.1, 0.3); k = int(s * SR); i = int(u * SR)
                if i + k < n:
                    pause[i:i + k] += rng.standard_normal(k) * 10 ** ((lv + rng.uniform(-4, 3)) / 20)
                speech.append((u, u + s)); u += s + rng.uniform(0.03, 0.15)
            end = u
            # a train beginning just inside the reach
            if rng.random() < 0.6:
                v = end + rng.uniform(2.3, 3.0); el = rng.uniform(0.04, 0.15); g = rng.uniform(0.2, 0.39); tl = rng.uniform(-50, -36)
                for _ in range(rng.integers(2, 10)):
                    k = int(el * SR); i = int(v * SR)
                    if i + k < n:
                        pause[i:i + k] += rng.standard_normal(k) * 10 ** (tl / 20)
                    v += el + g
                end = v
        elif r < 0.6:  # word
            s = rng.uniform(0.1, 0.45); lv = rng.uniform(-50, -36); k = int(s * SR); i = int(t * SR)
            if i + k < n:
                pause[i:i + k] += rng.standard_normal(k) * 10 ** (lv / 20)
            speech.append((t, t + s)); end = t + s
        else:  # breath
            s = rng.uniform(0.08, 0.45); lv = rng.uniform(-52, -40); k = int(s * SR); i = int(t * SR)
            if i + k < n:
                pause[i:i + k] += rng.standard_normal(k) * 10 ** (lv / 20)
            end = t + s
        t = end + rng.uniform(0.1, 4.5)
    first, last = lib.turn(40, -20, floor_db=-55), lib.turn(40, -20, floor_db=-55)
    wav = np.concatenate([first, pause.astype(np.float32), last]); off = first.size / SR
    for kind, fn in (("split", lib.loud_split), ("retime", lib.loud_retime)):
        m, p = fn(lib.MAIN, wav), fn(lib.PR, wav)
        only = m & ~p
        if only.any():
            sp = sum(int(only[int((off + a) / 0.02): int((off + b) / 0.02)].sum()) for a, b in speech)
            bad[kind].append((case, int(only.sum()), sp, [(round(a - off, 2), round(b - off, 2)) for a, b in lib.frames_runs(only)][:4]))
    # plan: speech wholly in a range on main but not on PR
    rm, rp = lib.plan(lib.MAIN, wav).ranges, lib.plan(lib.PR, wav).ranges
    def whole(rg, a, b):
        A, B = (off + a) * SR, (off + b) * SR
        return any(x <= A and B <= y for x, y in rg)
    w = [(round(a, 2), round(b, 2)) for a, b in speech if whole(rm, a, b) and not whole(rp, a, b)]
    if w:
        worse_plan.append((case, w))
print(f"minsil={MINSIL} seed={SEED} N={N}: main-only loud: split {len(bad['split'])} retime {len(bad['retime'])}; syllables whole on main not PR: {len(worse_plan)} cases")
for k in bad:
    for b in bad[k][:5]:
        print(" ", k, b)
for w in worse_plan[:5]:
    print("  plan", w)
