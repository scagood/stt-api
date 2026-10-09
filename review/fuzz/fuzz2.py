import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from parakeet_service import chunker
SR = 16000
def run(speech, total, t, mx, ctx, mn=20):
    chunker._speech_segments = lambda _w: speech
    return chunker.plan_chunks(np.broadcast_to(np.float32(0), (total,)), target_sec=t, max_sec=mx, min_sec=mn, context_sec=ctx)
def gen(rng, seconds, kind):
    sp, at = [], rng.uniform(0, 4)
    while True:
        if kind == 'ordinary': L = rng.uniform(0.3, 8); P = rng.uniform(0.1, 2.0)
        elif kind == 'long': L = rng.choice([rng.uniform(0.3, 3), rng.uniform(15, 45)]); P = rng.uniform(0.1, 1.0)
        else: L = rng.uniform(0.2, 25); P = rng.uniform(0.05, 4.0)
        if at + L > seconds: break
        sp.append((int(at*SR), int((at+L)*SR))); at += L + P
    return sp, int(seconds*SR)
rng = np.random.default_rng(int(sys.argv[2]))
avoid = {}; ex = {}
for trial in range(int(sys.argv[3])):
    kind = ['ordinary', 'long', 'mixed'][trial % 3]
    for (t, mx, ctx) in [(25, 30, 5), (60, 75, 5), (25, 30, 0)]:
        speech, total = gen(rng, rng.uniform(80, 400), kind)
        if not speech or total <= mx*SR: continue
        own = int((mx-2*ctx)*SR)
        r = run(speech, total, t, mx, ctx).ranges
        for s, e in r[:-1]:
            for a, b in speech:
                if a < e < b and b - a <= own:  # cut inside a phrase that would fit a chunk on its own
                    key = (t, mx, ctx, kind); avoid[key] = avoid.get(key, 0) + 1
                    if key not in ex: ex[key] = (speech, r, e)
print(avoid)
for k, (speech, r, c) in ex.items():
    print(k, 'cut', c/SR)
    i = min(range(len(speech)), key=lambda j: abs(speech[j][0]-c))
    print('  speech near', [(round(a/SR,2), round(b/SR,2)) for a, b in speech[max(0,i-4):i+3]])
    j = min(range(len(r)), key=lambda j: abs(r[j][1]-c))
    print('  ranges near', [(round(a/SR,2), round(b/SR,2)) for a, b in r[max(0,j-3):j+3]])
