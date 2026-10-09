"""Head: why does a cut move from a gap (main) into a word? Scores by head's own rule."""
import os, sys, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from synth import *

gap_db = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
cname = sys.argv[2] if len(sys.argv) > 2 else "whisper"
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
cfg = CONFIGS[cname]
rng = random.Random(f"k80a|1|{cname}|{scale}")
nrng = np.random.default_rng(abs(hash((1, cname, scale, gap_db, (50.0, 350.0), 0.0))) % 2**32)
F = PR.FRAME


def scores(wav, rstart, rend, point, at):
    frames = PR.frame_rms(wav[rstart:rend]).astype(np.float64) ** 2
    around = int(PR._LOCAL_SEC * SR) // F
    fr = (point - rstart) // F
    local = float(np.median(frames[max(0, fr - around): fr + around + 1])) + 1e-12
    p = lambda c, span: float(np.mean(wav[c - span // 2: c + span // 2].astype(np.float64) ** 2)) / local
    return p(point, PR._QUIET_SPAN), p(point, PR._DIP_SPAN), p(at, PR._QUIET_SPAN), local


rows = []
shown = 0
for it in range(250):
    segs_s, total_s = layout(rng, scale)
    if total_s <= cfg["mx"]:
        continue
    total = int(round(total_s * SR)); segs = [(int(round(a * SR)), int(round(b * SR))) for a, b in segs_s]
    wav, words = synth(segs, total, nrng, gap_db, tuple(float(x) for x in os.environ.get("GAPMS", "50,350").split(",")))
    starts = np.array([w[0] for w in words])
    rm, _ = plan(MAIN, wav, segs, cfg); rp, _ = plan(PR, wav, segs, cfg)
    fm, fp = forced_cuts(rm, segs), forced_cuts(rp, segs)
    if len(fm) != len(fp):
        continue
    for x, y in zip(fm, fp):
        if not in_word(x, starts, words) and in_word(y, starts, words):
            # the main range holding x: find PR range bounds around y (approx: main's packed range)
            r0 = max(a for a, b in rm if a < x)
            r1 = min(b for a, b in rm if b > x)
            # whole oversized range: walk adjacent ranges
            here, dip, lvl, local = scores(wav, max(0, x - 10 * SR), min(total, x + 10 * SR), x, y)
            j = np.searchsorted(starts, x, side="right") - 1
            gap = (words[j][1], words[j + 1][0]) if 0 <= j and j + 1 < len(words) else (words[j][1], words[j][1])
            i = np.searchsorted(starts, y, side="right") - 1
            ws, we = words[i]
            edge = min(x - gap[0], gap[1] - x) / SR
            gl = float(np.mean(wav[gap[0]:gap[1]].astype(np.float64) ** 2)) / local if gap[1] > gap[0] else -1
            wl = float(np.mean(wav[ws:we].astype(np.float64) ** 2)) / local
            rows.append((here, dip, lvl, edge, (gap[1] - gap[0]) / SR, gl, wl, abs(y - x) / SR))
            if shown < 6:
                shown += 1
                print(f"main {x/SR:.3f} in gap {gap[0]/SR:.3f}-{gap[1]/SR:.3f} ({(gap[1]-gap[0])*1000/SR:.0f} ms, {edge*1000:.0f} ms from its edge, gap pow {gl:.2f}) here200 {here:.2f} dip80 {dip:.2f} | head {y/SR:.3f} in word {ws/SR:.3f}-{we/SR:.3f} level200 {lvl:.2f}, word pow {wl:.2f}")
r = np.array(rows)
if len(r):
    print(f"n={len(r)}  here200>=0.5 & dip80>=0.5 (as head sees, approx local): {np.mean((r[:,0]>=0.5)&(r[:,1]>=0.5)):.0%}; "
          f"median distance to gap edge {np.median(r[:,3])*1000:.0f} ms; share < 40 ms {np.mean(r[:,3] < 0.04):.0%}; gap len median {np.median(r[:,4])*1000:.0f} ms; "
          f"gap pow median {np.median(r[:,5]):.2f}; dest word pow median {np.median(r[:,6]):.2f}; dest word quieter than the gap: {np.mean(r[:,6] < r[:,5]):.0%}; moved median {np.median(r[:,7]):.2f}s")
