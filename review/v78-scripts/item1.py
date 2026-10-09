import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/v78-scripts")
from lib import *
for seed in (0, 1, 2):
    wav = build(20, [(45.0, 1.0, -40), (48.95, 0.4, -40)], seed=seed)
    print("seed", seed)
    for n in ("main", "old", "head"):
        p = plan(n, wav, 0.0)
        inpause = [r for r in secs(p.ranges) if r[1] > 40.5 and r[0] < 59.5]
        print(f"  {n:5s} ranges {secs(p.ranges)}")
        word = (48.95, 49.35)
        cov = sum(max(0, min(b, word[1]) - max(a, word[0])) for a, b in [(a/SR, b/SR) for a, b in p.ranges])
        ps = [(round(a,2), round(b,2)) for a, b in pauses(n, wav) if b > 40 and a < 60]
        inside = [x for x in ps if x[0] > word[0] and x[0] < word[1]] + [x for x in ps if x[1] > word[0] and x[1] < word[1]]
        print(f"        word covered {cov:.3f}/0.400 s; pauses in 40-60: {ps}; pause edge inside word: {inside}")
