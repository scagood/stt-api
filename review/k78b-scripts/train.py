"""A quiet phrase, then a train of short sounds (footsteps / keystrokes / panting)
starting inside the phrase's 3 s reach. How far past the reach is kept, and is the
rest of the pause still cut out?"""
import sys
import numpy as np
import lib

PAUSE = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
res = []
for el in (0.06, 0.08, 0.1, 0.15, 0.2):
    for gap in (0.25, 0.3, 0.35, 0.38):
        for n in range(2, 14):
            for lvl in (-45, -42, -40):
                phrase = (1.0, 1.0, -40)            # quiet phrase 1.0-2.0 s into the pause
                t0 = 4.7                              # train starts 2.7 s after phrase end (inside 3 s reach)
                sounds = [phrase] + [(t0 + i * (el + gap), el, lvl) for i in range(n)]
                end_train = t0 + n * el + (n - 1) * gap
                if end_train > PAUSE - 0.5:
                    continue
                wav, off = lib.pause_between(PAUSE, sounds)
                out = []
                for mod in (lib.MAIN, lib.PR):
                    loud = lib.loud_split(mod, wav)
                    lr = lib.loud_retime(mod, wav)
                    i0, i1 = int(off / 0.02), int((off + PAUSE) / 0.02)
                    idx = np.flatnonzero(loud[i0:i1])
                    last = idx.max() * 0.02 if idx.size else None
                    idr = np.flatnonzero(lr[i0:i1])
                    lastr = idr.max() * 0.02 if idr.size else None
                    rg = lib.plan(mod, wav).ranges
                    # pause-time seconds decoded inside the pause
                    dec = sum(max(0, min(b / lib.SR, off + PAUSE) - max(a / lib.SR, off)) for a, b in rg)
                    out.append((last, lastr, round(dec, 2), len(rg)))
                res.append(((el, gap, n, lvl, round(end_train, 2)), out))
# biggest extension past the reach (phrase end 2.0 + 3.0 = 5.0 s)
def ext(o):
    return (o[0] or 0) - 5.0
diff = [(k, o) for k, o in res if o[0] != o[1]]
print("cases", len(res), "differ", len(diff))
diff.sort(key=lambda r: -((r[1][1][0] or 0) - (r[1][0][0] or 0)))
for k, o in diff[:12]:
    print(k, "main last loud/split %.2f retime %s dec %.2f n=%d" % (o[0][0] or -1, o[0][1], o[0][2], o[0][3]),
          "| PR last loud/split %.2f retime %s dec %.2f n=%d" % (o[1][0] or -1, o[1][1], o[1][2], o[1][3]))
more = [(k, o) for k, o in res if o[1][2] > o[0][2] + 0.01]
print("cases where PR decodes more of the pause:", len(more), "max extra s:", max((o[1][2] - o[0][2]) for k, o in more) if more else 0)
for k, o in sorted(more, key=lambda r: -(r[1][1][2] - r[1][0][2]))[:8]:
    print("  ", k, "main dec", o[0][2], "PR dec", o[1][2])
less = [(k, o) for k, o in res if o[1][2] < o[0][2] - 0.01]
print("cases where PR decodes less:", len(less))
