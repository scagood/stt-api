import json, sys, bisect
SR = 16000
CFG = {'v2c5': (25, 30, 20, 5), 'v2c0': (25, 30, 20, 0), 'v3c5': (60, 75, 20, 5), 'v3c0': (60, 75, 20, 0)}
def metrics(segs, total, ranges, windows, cfg):
    tg, mx, mn, cx = cfg
    seg = [(int(round(a * SR)), int(round(b * SR))) for a, b in segs]
    tot = int(round(total * SR))
    maximum = max(int(tg*SR), int(mx*SR))
    own = max(1, maximum - 2*int(cx*SR)) if tot > maximum else tot
    bad = []
    for i, (a, b) in enumerate(ranges):
        if not (0 <= a < b <= tot): bad.append('bounds')
        if tot > maximum and b - a > own: bad.append('own')
        if i and ranges[i-1][1] > a: bad.append('order')
    for (a, b), (wa, wb) in zip(ranges, windows):
        if not (wa <= a and b <= wb): bad.append('wincontain')
        if wb - wa > maximum: bad.append('winmax')
    # speech lost
    lost = 0
    for s, e in seg:
        cov = 0
        for a, b in ranges:
            cov += max(0, min(b, e) - max(a, s))
        lost += (e - s) - cov
    pts = set()
    for a, b in ranges:
        pts.add(a); pts.add(b)
    cuts = sum(1 for p in pts if any(s < p < e for s, e in seg))
    silent = sum(1 for a, b in ranges if not any(min(b, e) > max(a, s) for s, e in seg))
    short = sum(1 for a, b in ranges if b - a < 2*SR)
    return dict(n=len(ranges), cuts=cuts, lost=lost, bad=bad, silent=silent, short=short)
def load(v, seed):
    return json.load(open(f'out_{v}_{seed}.json'))
if __name__ == '__main__':
    A, B = sys.argv[1], sys.argv[2]
    for seed in sys.argv[3].split(','):
        lays = json.load(open(f'lay{seed}.json'))
        ra, rb = load(A, seed), load(B, seed)
        for name, cfg in CFG.items():
            st = dict(chunked=0, cut_better=0, cut_worse=0, n_more=0, n_fewer=0, totcutA=0, totcutB=0, silentA=0, silentB=0, shortA=0, shortB=0, lostA=0, badA=0)
            worse = []
            for i, (segs, total) in enumerate(lays):
                (raA, wa), (raB, wb) = ra[name][i], rb[name][i]
                ma, mb = metrics(segs, total, raA, wa, cfg), metrics(segs, total, raB, wb, cfg)
                if len(raA) > 1 or len(raB) > 1: st['chunked'] += 1
                st['totcutA'] += ma['cuts']; st['totcutB'] += mb['cuts']
                st['silentA'] += ma['silent']; st['silentB'] += mb['silent']
                st['shortA'] += ma['short']; st['shortB'] += mb['short']
                st['lostA'] += ma['lost'] > 0; st['badA'] += bool(ma['bad'])
                if ma['cuts'] < mb['cuts']: st['cut_better'] += 1
                if ma['cuts'] > mb['cuts']: st['cut_worse'] += 1; worse.append(i)
                if ma['n'] > mb['n']: st['n_more'] += 1
                if ma['n'] < mb['n']: st['n_fewer'] += 1
            print(seed, name, st, 'worse idx:', worse[:6])
