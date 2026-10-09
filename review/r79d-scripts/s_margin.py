"""Margin attacks on a one-piece clip (main never redoes these) and a two-piece clip
whose range redo is rejected.

WT=<worktree> python s_margin.py"""
import json
import logging

import lib
from lib import w, filler, run, names, check

logging.disable(logging.CRITICAL)


def one_piece(truth, beh, total=40.0):
    return run(total, [(0.0, total)], [(0.0, total)], None, lib.Model(truth, beh), vad=[(0.0, total)])


def report(tag, r, truth_names):
    out = names(r.words)
    lost = [x for x in truth_names if x not in out]
    extra = [x for x in out if x not in truth_names]
    dup = sorted({x for x in out if out.count(x) > 1})
    print(json.dumps(dict(tag=tag, n=len(out), extra=extra, dup=dup, lost_n=len(lost),
                          calls=[c for c in r.calls if c[0] != "first"], **check(r.words, r.text))))


# 1. A word the piece hears as one token, the redo as two, in the margin before the stretch.
for d in (0.16, 0.24, 0.32, 0.4):
    truth = filler(0.5, 19.6, 0.5, "a") + [w("into", 19.6)] + filler(20.2, 40.0, 0.5, "b")
    idx = next(i for i, x in enumerate(truth) if x.name == "into")

    def beh(a, b, phase, i, d=d, idx=idx):
        if phase == "first":
            return dict(skip=[(20.0, 30.0)])
        return dict(alt={idx: [(" in", 0.0), (" to", d)]})

    r = one_piece(truth, beh)
    report(f"split-before d={d}", r, [x.name for x in truth])

# 2. Same in the margin after the stretch.
for d in (0.16, 0.24, 0.32, 0.4):
    truth = filler(0.5, 30.4, 0.5, "a") + [w("into", 30.6)] + filler(31.2, 40.0, 0.5, "b")
    idx = next(i for i, x in enumerate(truth) if x.name == "into")

    def beh(a, b, phase, i, d=d, idx=idx):
        if phase == "first":
            return dict(skip=[(20.0, 30.3)])
        return dict(alt={idx: [(" in", 0.0), (" to", d)]})

    r = one_piece(truth, beh)
    report(f"split-after d={d}", r, [x.name for x in truth])

# 3. A number: the piece hears "1984" (digit tokens quickly), the redo "nineteen eighty-four".
for d in (0.24, 0.32, 0.48):
    truth = filler(0.5, 19.0, 0.5, "a") + [lib.W([(" ", 19.1), ("1", 19.1), ("9", 19.18), ("8", 19.26), ("4", 19.34)])] + filler(19.9, 40.0, 0.5, "b")
    idx = 37

    def beh(a, b, phase, i, d=d, idx=idx):
        if phase == "first":
            return dict(skip=[(21.0, 30.0)])
        return dict(alt={idx: [(" nine", 0.0), ("teen", 0.08), (" eight", d), ("y", d + 0.08), ("-", d + 0.16), ("four", d + 0.24)]})

    assert truth[idx].name == "1984", truth[idx].name
    r = one_piece(truth, beh)
    report(f"number d={d}", r, [x.name for x in truth])

# 4. Fragments at the redo's input edges (multi-token words straddling the margin start/end).
truth = []
t = 0.5
k = 0
while t < 40:
    truth.append(w(f"long{k}word", round(t, 3), ntok=4, step=0.1))
    t += 0.55
    k += 1


def beh(a, b, phase, i):
    if phase == "first":
        return dict(skip=[(20.0, 30.0)])
    return dict(frag=True)


r = one_piece(truth, beh)
report("fragments 4-token words every 0.55 s", r, [x.name for x in truth])
for x in r.words:
    if x["word"] not in [y.name for y in truth]:
        print("   invented", x)
