"""One continuous speech segment 3..37 s in 40 s of audio (v2 ctx5): how often
does the PR give up the lead / tail margin main keeps?"""
from h import *
cfg = CONFIGS["v2 ctx5"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
L0, L1, TOT = (float(x) for x in (sys.argv[2:5] if len(sys.argv) > 4 else (3, 37, 40)))
segs = [(int(L0 * SR), int(L1 * SR))]
total = int(TOT * SR)
shrink_lead = shrink_tail = 0
lost = []
for seed in range(N):
    rng = np.random.default_rng(seed)
    wav, words = synth(rng, segs, total)
    out = {k: plan(m, wav, segs, cfg) for k, m in M.items()}
    (rm, wm), (rp, wp) = out["main"], out["pr"]
    cm, cp = covered(wm, total), covered(wp, total)
    gone = cm & ~cp
    lead = (rp[0][0] - rm[0][0]) / SR
    tail = (rm[-1][1] - rp[-1][1]) / SR
    shrink_lead += lead > 0
    shrink_tail += tail > 0
    lost.append(gone.sum() / SR)
    if seed < 5:
        print(seed, "main", sec(rm), "pr", sec(rp), "lead gave", lead, "tail gave", tail, "audio main decoded, pr not: %.2fs" % (gone.sum() / SR))
lost = np.array(lost)
print(f"N={N}: lead shrank in {shrink_lead}, tail shrank in {shrink_tail}; layouts losing audio main decoded: {(lost > 0).sum()}, mean {lost[lost > 0].mean() if (lost > 0).any() else 0:.2f}s, max {lost.max():.2f}s")
