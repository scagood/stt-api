import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import MAIN, HEAD, SR, plan
from gens import gen_a
side, lq, gdb, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
shown = 0
for seed in range(N):
    r = np.random.default_rng([seed, int(-lq), gdb, 1 if side == "lead" else 2])
    total = 40 * SR
    wav, words = gen_a([(3 * SR, 37 * SR)], total, r)
    Q = r.uniform(1.0, 2.7)
    qs = (0, int(Q * SR)) if side == "lead" else (total - int(Q * SR), total)
    q, qw = gen_a([qs], total, r, gap_db=gdb, base_db=lq, room_db=-120)
    m = np.zeros(total, bool); m[qs[0]:qs[1]] = True
    wav[m] = wav[m] * 0 + q[m] + (r.standard_normal(m.sum()).astype(np.float32) * 10 ** (-65 / 20))
    ps = {nm: plan(mod, wav, "v2 ctx5", gate=None) for nm, mod in (("m", MAIN), ("h", HEAD))}
    cov = {}
    for nm, p in ps.items():
        c = np.zeros(total, bool)
        for a, b in p.ranges: c[a:b] = True
        cov[nm] = c
    lost = [(a / SR, b / SR) for a, b in qw if cov["m"][a:b].all() and not cov["h"][a:b].all()]
    if lost and shown < 3:
        shown += 1
        print(f"seed {seed}: quiet talker {qs[0]/SR:.2f}-{qs[1]/SR:.2f}s @ {lq} dBFS, gaps {gdb} dB down; VAD {[(round(s/SR,2), round(e/SR,2)) for s,e in ps['m'].speech]}")
        print(f"  main ranges {[(round(a/SR,2), round(b/SR,2)) for a,b in ps['m'].ranges]}")
        print(f"  head ranges {[(round(a/SR,2), round(b/SR,2)) for a,b in ps['h'].ranges]}  windows {[(round(a/SR,2), round(b/SR,2)) for a,b in ps['h'].windows]}")
        print(f"  words main decodes, head does not: {[(round(a,2), round(b,2)) for a,b in lost]}")
        st, fi = ps['m'].ranges[0][0], ps['m'].speech[0][0]
        rms = MAIN.frame_rms(wav[st:fi]); fl = np.percentile(rms, 10)
        snd = HEAD._sounds(wav, st, fi)
        print(f"  margin {st/SR:.2f}-{fi/SR:.2f}: floor (10th pct) {20*np.log10(fl):.1f} dBFS, loudest frame {20*np.log10(rms.max()):.1f} dBFS, frames counted as sound {int(snd.sum())}/{snd.size}")
