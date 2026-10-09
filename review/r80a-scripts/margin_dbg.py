import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import MAIN, HEAD, SR, plan
from gens import gen_a, gen_b, quiet_word
gen, side, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
G = gen_a if gen == "a" else gen_b
for seed in range(N):
    r = np.random.default_rng(10_000 + seed * 7 + (0 if gen == "a" else 1) + (0 if side == "tail" else 2))
    if side == "tail":
        lead = r.uniform(0.3, 3.0); L = r.uniform(36.0, 39.6); sp = (lead, lead + L)
        off = r.uniform(0.25, 2.9); wl = r.uniform(0.2, 0.45)
        w = (sp[1] + off, sp[1] + off + wl); total_s = w[1] + r.uniform(0.0, 3.0)
    else:
        lead = r.uniform(1.0, 3.5); L = r.uniform(36.0, 39.6)
        wl = r.uniform(0.2, 0.45); off = r.uniform(0.25, min(2.9, lead - wl))
        w = (lead - off - wl, lead - off); sp = (lead, lead + L); total_s = sp[1] + r.uniform(0.2, 3.0)
    lvl = r.uniform(-48, -38)
    total = int(total_s * SR)
    wav, words = G([(int(sp[0] * SR), int(sp[1] * SR))], total, r)
    a, b = int(w[0] * SR), int(w[1] * SR)
    quiet_word(wav, a, b, lvl, r, gen)
    pm = plan(MAIN, wav, "v2 ctx5", gate=None); ph = plan(HEAD, wav, "v2 ctx5", gate=None)
    cm = np.zeros(total, bool); ch = np.zeros(total, bool)
    for r0, r1 in pm.ranges: cm[r0:r1] = True
    for r0, r1 in ph.ranges: ch[r0:r1] = True
    d = cm & ~ch
    big = False
    if d.any():
        idx = np.flatnonzero(d); seg = wav[idx[0]:idx[-1]+1]; rr = MAIN.frame_rms(seg)
        marg = wav[pm.speech[-1][1]: pm.ranges[-1][1]] if side == "tail" else wav[pm.ranges[0][0]: pm.speech[0][0]]
        mr = MAIN.frame_rms(marg); fl = float(np.percentile(mr, 10)) if mr.size else 1e-6
        over = 20*np.log10((rr.max() if rr.size else 1e-9)/fl); big = over > 12
        if big: pass
    if d[a:b].any():
        print(f"seed {seed} {side}: speech {sp[0]:.2f}-{sp[1]:.2f} word {w[0]:.3f}-{w[1]:.3f} @{lvl:.1f} total {total_s:.2f}")
        print(f"  VAD {[(round(s/SR,3), round(e/SR,3)) for s,e in pm.speech]}")
        print(f"  main {[(round(x/SR,3), round(y/SR,3)) for x,y in pm.ranges]} win {[(round(x/SR,3), round(y/SR,3)) for x,y in pm.windows]}")
        print(f"  head {[(round(x/SR,3), round(y/SR,3)) for x,y in ph.ranges]} win {[(round(x/SR,3), round(y/SR,3)) for x,y in ph.windows]}")
        idx = np.flatnonzero(d); print(f"  given up {idx[0]/SR:.3f}-{(idx[-1]+1)/SR:.3f} overlap word {d[a:b].sum()/SR:.3f}s")
