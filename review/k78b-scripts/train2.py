import sys
import numpy as np
import lib
PAUSE = float(sys.argv[1])
el, gap, lvl = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
for n in range(1, 16):
    t0 = 4.7
    sounds = [(1.0, 1.0, -40)] + [(t0 + i * (el + gap), el, lvl) for i in range(n)]
    end_train = t0 + n * el + (n - 1) * gap
    if end_train > PAUSE:
        break
    wav, off = lib.pause_between(PAUSE, sounds)
    row = []
    for mod in (lib.MAIN, lib.PR):
        loud = lib.loud_split(mod, wav)
        i0, i1 = int(off / 0.02), int((off + PAUSE) / 0.02)
        rr = [(round(a, 2), round(b, 2)) for a, b in lib.frames_runs(loud[i0:i1])]
        rg = lib.plan(mod, wav).ranges
        dec = sum(max(0, min(b / lib.SR, off + PAUSE) - max(a / lib.SR, off)) for a, b in rg)
        lr = lib.loud_retime(mod, wav)
        pz = [(round(a - off, 2), round(b - off, 2)) for a, b in mod[1].pauses(wav) if b > off and a < off + PAUSE]
        row.append((rr[-1][1] if rr else None, round(dec, 2), len(rg), pz))
    print(n, "train end %.2f" % end_train, "| main last loud", row[0][0], "dec", row[0][1], "chunks", row[0][2], "| PR last loud", row[1][0], "dec", row[1][1], "chunks", row[1][2])
    if row[0][3] != row[1][3]:
        print("    retime pauses main", row[0][3][-3:], "\n    retime pauses PR  ", row[1][3][-3:])
