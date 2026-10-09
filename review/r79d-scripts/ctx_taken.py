"""A neighbour's *context* word (one it never keeps) heard differently from the
stretch redo's word at the same time: does the redo's word still go in?

WT=<worktree> python ctx_taken.py"""
import asyncio
import logging
from types import SimpleNamespace

import lib
from lib import routes

logging.disable(logging.CRITICAL)


def heard(origin, words):
    words = sorted(words, key=lambda x: x[1])
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
            a, b = round(p[0] / 16000, 3), round((p[0] + p.size) / 16000, 3)
            self.asked.append((a, b))
            out.append(self.fn(a, b))
        return out


left_words = filler("a", 0.5, 54.5) + filler("c", 55.0, 59.0)
right_words = filler("b", 60.5, 119.5)
prep = lib.prepared(120.0, [(0, 60), (60, 120)], [(0, 65), (55, 120)], [(0, 120)])

for case in ("L skips to the cut; R's context heard 'Holmes' at 59.5",
             "R skips from the cut; L's context heard 'Holmes' at 60.3"):
    if case.startswith("L"):
        truth = left_words + [(" Holmes", 59.5)] + right_words
        first = [heard(0, [x for x in truth if x[1] < 55.5]), heard(55, [x for x in truth if x[1] >= 55])]

        def fn(a, b):
            if (a, b) == (0.0, 60.0):
                return heard(0, [x for x in truth if x[1] < 55.5])
            if (a, b) == (60.0, 120.0):
                return heard(60, [x for x in truth if x[1] >= 60])
            return heard(a, [(" homes" if w == " Holmes" else w, t) for w, t in truth if a <= t < b])
    else:
        truth = left_words + [(" Holmes", 60.3)] + right_words
        first = [heard(0, [x for x in truth if x[1] < 65]), heard(55, [x for x in truth if 55 <= x[1] < 60.1 or x[1] >= 64.5])]

        def fn(a, b):
            if (a, b) == (0.0, 60.0):
                return heard(0, [x for x in truth if x[1] < 60])
            if (a, b) == (60.0, 120.0):
                return heard(60, [x for x in truth if x[1] >= 64.5])
            return heard(a, [(" homes" if w == " Holmes" else w, t) for w, t in truth if a <= t < b])
    w = Redo(fn)
    res = asyncio.run(routes._redo_stalled(lib.request(w), [prep], [r for r in first], "parakeet-v3:fp32"))
    text, segs, words = routes._stitch(prep, res)
    print(case, "| redos", w.asked, "|", [(x["word"], round(x["start"], 2)) for x in words if 58.9 <= x["start"] <= 61.1])
