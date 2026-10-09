import json, logging
import lib
from lib import w, filler, run, names, check
logging.disable(logging.CRITICAL)

def one_piece(truth, beh, total=40.0):
    return run(total, [(0.0, total)], [(0.0, total)], None, lib.Model(truth, beh), vad=[(0.0, total)])

def show(tag, r, lo, hi):
    print(tag, "calls", [c for c in r.calls if c[0] != "first"])
    print("   ", [(x["word"], round(x["start"], 2)) for x in r.words if lo <= x["start"] <= hi])
    print("   ", check(r.words, r.text))

# number in the margin before the stretch
for d in (0.24, 0.32, 0.48):
    truth = filler(0.5, 18.6, 0.5, "a") + [lib.W([(" ", 18.7), ("1", 18.7), ("9", 18.78), ("8", 18.86), ("4", 18.94)])] + filler(19.5, 40.0, 0.5, "b")
    idx = next(i for i, x in enumerate(truth) if x.name == "1984")
    def beh(a, b, phase, i, d=d, idx=idx):
        if phase == "first":
            return dict(skip=[(21.0, 30.0)])
        return dict(alt={idx: [(" nine", 0.0), ("teen", 0.08), (" eight", d), ("y", d + 0.08), ("-", d + 0.16), ("four", d + 0.24)]})
    r = one_piece(truth, beh)
    show(f"number d={d}", r, 18.0, 20.0)

# split word in the text: show it
truth = filler(0.5, 19.6, 0.5, "a") + [w("into", 19.6)] + filler(20.2, 40.0, 0.5, "b")
idx = next(i for i, x in enumerate(truth) if x.name == "into")
def beh(a, b, phase, i, idx=idx):
    if phase == "first":
        return dict(skip=[(20.0, 30.0)])
    return dict(alt={idx: [(" in", 0.0), (" to", 0.24)]})
r = one_piece(truth, beh)
show("split into -> in to (0.24)", r, 19.0, 20.6)
i = r.text.find("into")
print("   text:", r.text[i-20:i+30])

# fragment at the redo's input start: a single-token word whose sound runs past the redo start
truth = filler(0.5, 40.0, 0.5, "a")
def beh(a, b, phase, i):
    if phase == "first":
        return dict(skip=[(20.0, 30.0)])
    return dict(extras=[(" thing", a)])  # the tail of the word straddling the redo's start, heard as a word
r = one_piece(truth, beh)
show("fragment at redo start", r, 17.0, 19.0)
# and at its end
def beh(a, b, phase, i):
    if phase == "first":
        return dict(skip=[(20.0, 30.0)])
    return dict(extras=[(" uh", b - 0.08)])
r = one_piece(truth, beh)
show("made-up word at redo end (#68)", r, 31.0, 33.0)
