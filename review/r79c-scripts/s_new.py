"""New-rule probes on head: last redo word, neighbour context blocking, anchors.
WT=<worktree> python s_new.py"""
import logging
import lib
from lib import w, filler, run, names, check

logging.disable(logging.CRITICAL)


def show(tag, r, lo, hi, truth=None):
    ws = [(x["word"], round(x["start"], 2)) for x in r.words if lo <= x["start"] <= hi]
    extra = ""
    if truth is not None:
        tn = [x.name for x in truth]
        out = names(r.words)
        extra = f" lost={[x for x in tn if x not in out][:8]} n_lost={len([x for x in tn if x not in out])} dup={sorted({x for x in out if out.count(x) > 1})}"
    print(tag, "| calls", [c for c in r.calls if c[0] != "first"], "|", ws, check(r.words, r.text), extra)


# R1. one-piece clip, piece stops at 20 s of 30 s; speech runs to 29.5 s; redo hears all.
truth = filler(0.5, 29.6, 0.5, "a")
r = run(30.0, [(0, 30)], [(0, 30)], None, lib.Model(truth, lambda a, b, p, i: dict(skip=[(20.0, 30.0)]) if p == "first" else {}), vad=[(0, 30)])
show("R1 trailing skip, one piece: last word", r, 27.0, 30.0, truth)
# R1b: only two words in the trailing stretch
truth = filler(0.5, 20.0, 0.5, "a") + [w("Yes", 24.0), w("indeed", 27.0)]
r = run(30.0, [(0, 30)], [(0, 30)], None, lib.Model(truth, lambda a, b, p, i: dict(skip=[(20.0, 30.0)]) if p == "first" else {}), vad=[(0, 30)])
show("R1b trailing stretch with 2 words", r, 19.0, 30.0, truth)
# R1c: last piece of a two-piece clip stops at 100 s of 120; main's range redo skips again
ranges, windows = lib.two(120.0, 60.0)
truth = filler(0.5, 119.6, 0.5, "a")
beh = lambda a, b, p, i: dict(skip=[(100.0, 120.0)]) if p in ("first", "_redo_ranges") else {}
r = run(120.0, ranges, windows, [(0, 120)], lib.Model(truth, beh))
show("R1c last piece trailing skip, range redo skips again", r, 117.0, 120.0, truth)

# R2. right piece's context word (before the cut, in the left range) heard differently than the left's stretch redo
# left piece skips 52-60 (to its range end), whole and range redo; right piece's context heard "homes" at 59.7
truth = filler(0.5, 59.5, 0.5, "a") + [w("Holmes", 59.7)] + filler(60.3, 120, 0.5, "b")
hi_ = next(i for i, x in enumerate(truth) if x.name == "Holmes")
def beh(a, b, phase, i):
    if phase in ("first", "_redo_ranges") and a < 30:
        return dict(skip=[(52.0, 60.0)])
    if phase == "first" and a > 30:
        return dict(alt={hi_: [(" homes", 0.0)]})
    return {}
r = run(120.0, ranges, windows, [(0, 120)], lib.Model(truth, beh))
show("R2 right piece heard 'homes' in its context; left redo 'Holmes'", r, 58.5, 61.0, truth)

# R3. stray word splits a skip: wordless 2.6 s gap before it is inside the redo window, outside the anchors
truth = filler(0.2, 120, 0.5, "w")
beh = lambda a, b, p, i: dict(skip=[(30.0, 32.6), (32.8, 45.0)]) if p in ("first", "_redo_ranges") else {}
r = run(120.0, ranges, windows, [(0, 120)], lib.Model(truth, beh))
show("R3 stray word at 32.7", r, 29.0, 34.0, truth)
