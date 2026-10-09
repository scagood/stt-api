import sys, importlib, numpy as np
sys.path.insert(0, sys.argv[1])  # worktree
from parakeet_service import chunker
SR = 16000

def run(speech, total, target, mx, mn, ctx):
    chunker._speech_segments = lambda _w: speech
    return chunker.plan_chunks(np.broadcast_to(np.float32(0), (total,)), target_sec=target, max_sec=mx, min_sec=mn, context_sec=ctx)

def gen(rng, seconds, kind):
    sp, at = [], rng.uniform(0, 4)
    while True:
        if kind == 'ordinary': L = rng.uniform(0.3, 8); P = rng.uniform(0.1, 2.0)
        elif kind == 'long': L = rng.choice([rng.uniform(0.3, 3), rng.uniform(15, 45)]); P = rng.uniform(0.1, 1.0)
        else: L = rng.uniform(0.2, 25); P = rng.uniform(0.05, 4.0)
        if at + L > seconds: break
        sp.append((int(at*SR), int((at+L)*SR))); at += L + P
    return sp, int(seconds*SR)

stats = dict(chunks=0, insp=0, silent=0, lost=0, tiny=0, over=0, bad_window=0)
examples = {}
rng = np.random.default_rng(int(sys.argv[2]))
for trial in range(int(sys.argv[3])):
    kind = ['ordinary', 'long', 'mixed'][trial % 3]
    for (t, mx, ctx) in [(25, 30, 5), (60, 75, 5), (25, 30, 0)]:
        speech, total = gen(rng, rng.uniform(40, 400), kind)
        if not speech: continue
        own_max = int((mx - 2*ctx)*SR)
        p = run(speech, total, t, mx, 20, ctx)
        r = p.ranges
        stats['chunks'] += len(r)
        cuts = [e for s, e in r[:-1]]
        stats['insp'] += sum(1 for c in cuts for s, e in speech if s < c < e)
        for (s, e) in r:
            if not any(a < e and s < b for a, b in speech):
                stats['silent'] += 1; examples.setdefault('silent', (speech, total, t, mx, ctx, r))
            if e - s > own_max: stats['over'] += 1
            if e - s < 1*SR and any(a < e and s < b for a, b in speech): stats['tiny'] += 1; examples.setdefault('tiny', (speech, total, t, mx, ctx, r))
        for a, b in speech:
            covered = sum(max(0, min(b, e) - max(a, s)) for s, e in r)
            if covered < b - a: stats['lost'] += 1; examples.setdefault('lost', (speech, total, t, mx, ctx, r))
        for (s, e), (ws, we) in zip(r, p.windows):
            if not (ws <= s < e <= we and we - ws <= int(mx*SR)): stats['bad_window'] += 1
print(stats)
for k, v in examples.items():
    speech, total, t, mx, ctx, r = v
    print(k, 'model', t, mx, ctx, 'total', total/SR)
    print('  speech', [(round(a/SR,2), round(b/SR,2)) for a, b in speech][:40])
    print('  ranges', [(round(a/SR,2), round(b/SR,2)) for a, b in r][:40])
