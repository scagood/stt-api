import json, logging
import lib
from lib import w, filler, run, names, check
logging.disable(logging.CRITICAL)

def one_piece(truth, beh, total=40.0):
    return run(total, [(0.0, total)], [(0.0, total)], None, lib.Model(truth, beh), vad=[(0.0, total)])

for d in (0.16, 0.24, 0.32, 0.48):
  for where in ("before", "after"):
    if where == "before":
        truth = filler(0.5, 20.5, 0.5, "a") + [lib.W([(" ", 20.6), ("1", 20.6), ("9", 20.68), ("8", 20.76), ("4", 20.84)])] + filler(21.5, 40.0, 0.5, "b")
        skip = (22.4, 31.0)
    else:
        truth = filler(0.5, 31.5, 0.5, "a") + [lib.W([(" ", 31.6), ("1", 31.6), ("9", 31.68), ("8", 31.76), ("4", 31.84)])] + filler(32.5, 40.0, 0.5, "b")
        skip = (20.0, 31.2)
    idx = next(i for i, x in enumerate(truth) if x.name == "1984")
    def beh(a, b, phase, i, d=d, idx=idx, skip=skip):
        if phase == "first":
            return dict(skip=[skip])
        return dict(alt={idx: [(" nine", 0.0), ("teen", 0.08), (" eight", d), ("y", d + 0.08), ("-", d + 0.16), ("four", d + 0.24)]})
    r = one_piece(truth, beh)
    at = truth[idx].t
    print(f"number {where} d={d}", [c for c in r.calls if c[0] != "first"], [(x["word"], round(x["start"], 2)) for x in r.words if at - 0.6 <= x["start"] <= at + 1.0], check(r.words, r.text))
