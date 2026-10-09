"""Why does #80 move a cut from a gap into a word? (10 dB gaps)"""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from synth import *

gap_db = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
cfg = CONFIGS["v2 ctx5"]
rng = random.Random("diag"); nrng = np.random.default_rng(5)
shown = 0
stats = []
for it in range(300):
    segs_s, total_s = layout(rng, 1.6)
    if total_s <= cfg["mx"]:
        continue
    total = int(round(total_s * SR)); segs = [(int(round(a * SR)), int(round(b * SR))) for a, b in segs_s]
    wav, words = synth(segs, total, nrng, gap_db, (50, 350))
    starts = np.array([w[0] for w in words])
    rm, _ = plan(MAIN, wav, segs, cfg); rp, _ = plan(PR, wav, segs, cfg)
    fm, fp = forced_cuts(rm, segs), forced_cuts(rp, segs)
    if len(fm) != len(fp):
        continue
    # median as _quiet_cuts sees it: per oversized range; approximate with the segment holding the cut
    for x, y in zip(fm, fp):
        a, b = in_word(x, starts, words), in_word(y, starts, words)
        if not a and b:
            i = np.searchsorted(starts, y, side="right") - 1
            ws, we = words[i]
            seg = next((s, e) for s, e in segs if s < y < e)
            med = float(np.median(MAIN.frame_rms(wav[seg[0]:seg[1]])) ** 2)
            p = lambda c: float(np.mean(wav[c - 640:c + 640].astype(np.float64) ** 2)) / med
            j = np.searchsorted(starts, x, side="right") - 1
            gap = (words[j][1], words[j + 1][0]) if j + 1 < len(words) else None
            wl = float(np.mean(wav[ws:we].astype(np.float64) ** 2)) / med
            stats.append((p(y), p(x), wl, (we - ws) / SR, (y - ws) / SR, ((gap[1] - gap[0]) / SR) if gap else -1))
            if shown < 12:
                shown += 1
                print(f"even {x/SR:.3f} in gap {gap[0]/SR:.3f}-{gap[1]/SR:.3f} ({(gap[1]-gap[0])*1000/SR:.0f} ms) pow {p(x):.3f} | #80 {y/SR:.3f} in word {ws/SR:.3f}-{we/SR:.3f} at +{(y-ws)/SR:.3f}s pow {p(y):.3f}, word mean pow {wl:.3f}")
s = np.array(stats)
if len(s):
    print(f"n={len(s)}: #80 point pow median {np.median(s[:,0]):.3f}; even (gap) pow median {np.median(s[:,1]):.3f}; share of even pow >= 0.25: {(s[:,1] >= 0.25).mean():.0%}; "
          f"word mean pow median {np.median(s[:,2]):.3f}; share word mean pow < 0.25 {(s[:,2] < 0.25).mean():.0%}; gap len median {np.median(s[:,5]):.3f}s, share gap < 80 ms {(s[:,5] < 0.08).mean():.0%}")
