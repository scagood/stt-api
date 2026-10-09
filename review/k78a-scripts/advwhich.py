import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/k78a-scripts")
from lib import *
import diff as D
cases = {1: [84, 197, 430], 2: [338, 569], 3: [89], 4: [458, 594], 5: [588], 6: [335, 597], 7: [261, 349, 375, 458, 550], 8: [49, 393, 94, 378]}
def status_plan(ranges, w):
    return D.score_plan(ranges, [w])
def status_rt(ps, w):
    return D.score_pauses(ps, [w])
for seed, its in cases.items():
    rng = np.random.default_rng(seed)
    want = set(its)
    for it in range(max(its) + 1):
        floor = float(rng.uniform(-62, -50)); P = float(rng.uniform(6, 30)); sounds = []
        t = 40 + float(rng.uniform(0.3, 3)); ph_end = None
        for _ in range(int(rng.integers(1, 4))):
            d = float(rng.uniform(0.5, 3.0)); lvl = float(rng.uniform(-36, -31))
            if t + d > 40 + P - 0.5: break
            sounds.append((t, d, lvl)); ph_end = t + d
            t += d + float(rng.uniform(0.1, 1.0))
        if ph_end is None: continue
        w0 = ph_end + 3.0 - float(rng.uniform(0.02, 0.4)); wd = float(rng.uniform(0.1, 0.48))
        sounds.append((w0, wd, float(rng.uniform(-40, -31))))
        t = w0 + wd + float(rng.uniform(0.05, 1.0))
        while t < 40 + P - 0.2:
            d = float(rng.uniform(0.1, 1.2)); lvl = floor + float(rng.uniform(6, 16))
            d = min(d, 40 + P - t); sounds.append((t, d, lvl))
            t += d + float(rng.uniform(0.05, 2.5))
        if it not in want: continue
        wav = build(P, sounds, seed=it + 100000 * seed, floor_db=floor)
        pm, ph = plan("main", wav, 0.0).ranges, plan("head", wav, 0.0).ranges
        rm, rh = pauses("main", wav), pauses("head", wav)
        out = []
        for s, d, l in sounds:
            if d < 0.12: continue
            w = (s, s + d)
            a, b = status_plan(pm, w), status_plan(ph, w)
            if a != b and b[0] == 0: out.append(f"plan: {s:.2f}-{s+d:.2f} +{l-floor:.1f}dB main{a} head{b}")
            if status_rt(rh, w) > status_rt(rm, w): out.append(f"retime: {s:.2f}-{s+d:.2f} ({d:.2f}s) +{l-floor:.1f}dB now in a pause")
        print(seed, it, "; ".join(out))
