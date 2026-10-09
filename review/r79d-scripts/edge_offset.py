"""The word where the piece's decode resumes after a skip ("Rome", 28.0 s), timed
d s apart by the stretch redo. One-piece 40 s clip, or two pieces cut at 60 s
with the skip in piece 0 (whose range redo skips again).

WT=<worktree> python edge_offset.py [one|two] [edge: far|near]"""
import asyncio
import logging
import sys
from types import SimpleNamespace

import lib
from lib import routes

logging.disable(logging.CRITICAL)
shape = sys.argv[1] if len(sys.argv) > 1 else "one"
edge = sys.argv[2] if len(sys.argv) > 2 else "far"


def heard(origin, words):
    return SimpleNamespace(text="".join(t for t, _ in words).strip(), tokens=[t for t, _ in words],
                           timestamps=[round(a - origin, 6) for _, a in words])


def filler(name, start, stop, step=0.5):
    return [(f" {name}{i}", start + step * i) for i in range(int(round((stop - start) / step)) + 1)]


class Redo:
    def __init__(self, fn):
        self.fn, self.asked = fn, []

    async def submit_many(self, pieces, key):
        out = []
        for p in pieces:
            a, b = p[0] / 16000, (p[0] + p.size) / 16000
            self.asked.append((round(a, 2), round(b, 2)))
            out.append(self.fn(a, b))
        return out


for d in (-0.64, -0.56, -0.48, -0.44, -0.4, -0.32, 0.0, 0.32, 0.48, 0.56, 0.64):
    if edge == "far":
        # resumes at "Rome" after skipping 20-28 s; redo times Rome at 28+d
        before = filler("a", 0.5, 19.5)
        after = [(" Rome", 28.0)] + filler("b", 28.5, 39.5)
        middle = filler("c", 20.0, 27.5)
        truth_first = before + after
        redo_words = lambda a, b, d=d: [(w, t) for w, t in before + middle + [(" Rome", 28.0 + d)] + after[1:] if a <= t < b]
        lo, hi = 26.8, 29.2
    else:
        # last word before the skip is "Rome" at 20.0; skipped 20.3-28; redo times Rome at 20+d
        before = filler("a", 0.5, 19.5) + [(" Rome", 20.0)]
        after = filler("b", 28.0, 39.5)
        middle = filler("c", 20.5, 27.5)
        truth_first = before + after
        redo_words = lambda a, b, d=d: [(w, t) for w, t in before[:-1] + [(" Rome", 20.0 + d)] + middle + after if a <= t < b]
        lo, hi = 18.8, 21.4
    if shape == "one":
        prep = lib.prepared(40.0, [(0, 40)], [(0, 40)], None)
        routes.speech_segments = lambda wav: [(0, wav.size)]
        first = [heard(0.0, truth_first)]
        w = Redo(lambda a, b: heard(a, sorted(redo_words(a, b), key=lambda x: x[1])))
    else:
        tail = filler("e", 40.0, 119.5)
        prep = lib.prepared(120.0, [(0, 60), (60, 120)], [(0, 65), (55, 120)], [(0, 120)])
        first = [heard(0.0, [x for x in truth_first + tail if x[1] < 65]), heard(55.0, [x for x in tail if x[1] >= 55])]

        def fn(a, b):
            if (a, b) == (0.0, 60.0):  # range redo skips again
                return heard(0.0, [x for x in truth_first + tail if x[1] < 60])
            return heard(a, sorted([x for x in redo_words(a, b)], key=lambda x: x[1]))
        w = Redo(fn)
    res = asyncio.run(routes._redo_stalled(lib.request(w), [prep], first, "parakeet-v3:fp32"))
    text, segs, words = routes._stitch(prep, res)
    near = [(x["word"], round(x["start"], 2)) for x in words if lo <= x["start"] <= hi]
    print(f"{shape} {edge} d={d:+.2f} redo {w.asked}: Rome x{[x['word'] for x in words].count('Rome')} {near}")
