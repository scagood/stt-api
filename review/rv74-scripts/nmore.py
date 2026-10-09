import json
from cmp import metrics, load
cfg = (25, 30, 20, 5)
SR=16000
for seed in (1, 2, 3):
    lays = json.load(open(f'lay{seed}.json'))
    R = {v: load(v, seed)['v2c5'] for v in ('rv74', 'rv74-main', 'rv74-base', 'rv74-old')}
    for i, (segs, total) in enumerate(lays):
        m = {v: metrics(segs, total, *R[v][i], cfg) for v in R}
        if m['rv74']['n'] > m['rv74-main']['n']:
            print(seed, i, total, {v: (m[v]['n'], m[v]['cuts']) for v in m})
            for v in R: print('  ', v, [(round(a/SR,3), round(b/SR,3)) for a,b in R[v][i][0]])
            print('   segs', segs)
