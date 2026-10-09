import asyncio, logging, time, json
from types import SimpleNamespace
import lib
from lib import w, filler, run, names, check, routes, S
logging.disable(logging.CRITICAL)

def show(tag, r, lo=None, hi=None):
    ws = [(x["word"], round(x["start"], 2)) for x in r.words if (lo is None or lo <= x["start"] <= hi)]
    print(tag, "| calls", [c for c in r.calls if c[0] != "first"], "|", ws, check(r.words, r.text))

# 1. Both sides of a cut skipped, each redo hears the cut word, differently.
truth = filler(0.5, 59.5, 0.5, "a") + [w("Holmes", 59.98)] + filler(60.6, 120, 0.5, "b")
hi_ = next(i for i, x in enumerate(truth) if x.name == "Holmes")
def beh(a, b, phase, i):
    if phase == "first":
        return dict(skip=[(52.0, 68.0)])
    if phase == "_redo_ranges":
        return dict(skip=[(52.0, 68.0)])
    # stretch redos: the left piece's (starts before 55) hears "Holmes", the right's "homes"
    if a < 55:
        return dict(jit=lambda k: -1 if k == hi_ else 0)
    return dict(alt={hi_: [(" homes", 0.0)]}, jit=lambda k: 1 if k == hi_ else 0)
ranges, windows = lib.two(120.0, 60.0)
r = run(120.0, ranges, windows, [(0, 120)], lib.Model(truth, beh))
show("both sides of cut, misheard by one", r, 59.0, 61.0)
def beh2(a, b, phase, i):
    if phase in ("first", "_redo_ranges"):
        return dict(skip=[(52.0, 68.0)])
    if a < 55:
        return dict(jit=lambda k: -1 if k == hi_ else 0)
    return dict(jit=lambda k: 1 if k == hi_ else 0)
r = run(120.0, ranges, windows, [(0, 120)], lib.Model(truth, beh2))
show("both sides of cut, same word", r, 59.0, 61.0)

# 2. Exactly two invented words in music, one-piece clip: main none, head both.
truth = filler(0.5, 20.0, 0.5, "a") + filler(32.0, 50.0, 0.5, "b")
def beh(a, b, phase, i):
    if phase == "first":
        return {}
    return dict(extras=[(" la", 24.0), (" la", 27.0)])
r = run(50.0, [(0, 50)], [(0, 50)], None, lib.Model(truth, beh), vad=[(0, 50)])
show("music 20-32 s, redo invents 'la la'", r, 19.0, 33.0)
def beh(a, b, phase, i):
    return dict(extras=[(" la", 24.0)]) if phase != "first" else {}
r = run(50.0, [(0, 50)], [(0, 50)], None, lib.Model(truth, beh), vad=[(0, 50)])
show("music, redo invents one 'la'", r, 19.0, 33.0)

# 3. Redo returns nothing / only margin words / piece with no words at all.
truth = filler(0.5, 40.0, 0.5, "a")
for tag, beh in [
    ("redo empty", lambda a, b, p, i: dict(skip=[(20, 30)]) if p == "first" else dict(skip=[(0, 100)])),
    ("redo only margins", lambda a, b, p, i: dict(skip=[(20, 30)])),
    ("redo one word in stretch", lambda a, b, p, i: dict(skip=[(20, 30)]) if p == "first" else dict(skip=[(20, 25), (25.6, 30)])),
]:
    r = run(40.0, [(0, 40)], [(0, 40)], None, lib.Model(truth, beh), vad=[(0, 40)])
    show(tag, r, 19.0, 21.0)
r = run(40.0, [(0, 40)], [(0, 40)], None, lib.Model(truth, lambda a, b, p, i: dict(skip=[(0, 100)]) if p == "first" else {}), vad=[(0, 40)])
show("piece heard nothing (one piece)", r, 0, 2)
ranges, windows = lib.two(120.0, 60.0)
truth = filler(0.5, 120, 0.5, "a")
r = run(120.0, ranges, windows, [(0, 120)], lib.Model(truth, lambda a, b, p, i: dict(skip=[(0, 66)]) if p != "_redo_stretches" else {}))
print("piece 0 heard nothing, range redo too:", len(r.words), "words; first", names(r.words)[:3], check(r.words, r.text), [c for c in r.calls if c[0] != "first"])

# 4. Exactly 3.0 s untimed stretch (inline vs pool); 2.99 s.
for gap in (2.99, 3.0, 3.01):
    toks = [" a", " b"]; ts = [1.0, 1.0 + 0.32 + gap]
    prep = lib.prepared(10.0, [(0, 10)], [(0, 10)], None)
    res = [SimpleNamespace(text="a b", tokens=toks, timestamps=ts)]
    print("untimed gap", gap, {k: [(a / 16000, b / 16000) for a, b in v] for k, v in routes._untimed(prep, res).items()})
