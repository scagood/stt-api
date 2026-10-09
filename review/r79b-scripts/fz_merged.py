"""Random-input fuzz of _merged, _kept_near, _redo_windows, _untimed: any exception?"""
import random, traceback, math
from types import SimpleNamespace
import lib
from lib import routes
SR = 16000
TOK = [" the", "▁the", " ", "▁", "ing", ",", ".", " 1", "9", "8", "4", " £", "'s", " ", "-", "?", " a", " no", " I", "Ā", " 中", "文", "", " .", "...", " -"]
def rand_result(rng, n, span, unsorted=False, bad=False):
    toks = [rng.choice(TOK) for _ in range(n)]
    ts = sorted(rng.uniform(0, span) for _ in range(n))
    if unsorted:
        rng.shuffle(ts)
    if bad:
        for k in range(n):
            r = rng.random()
            if r < 0.05: ts[k] = float("nan")
            elif r < 0.1: ts[k] = -1.0
            elif r < 0.12: ts[k] = None
            elif r < 0.14: ts[k] = float("inf")
        if rng.random() < 0.2: ts = ts[: rng.randrange(n + 1)]
    return SimpleNamespace(text="".join(toks), tokens=toks, timestamps=ts)
errors = 0
for case in range(30000):
    rng = random.Random(case)
    span = rng.uniform(1, 80)
    origin = rng.randrange(0, 10 * SR)
    res = rand_result(rng, rng.randrange(0, 60), span, rng.random() < 0.2, rng.random() < 0.2)
    start = origin + rng.randrange(0, int(span * SR))
    stop = start + rng.randrange(1, int(10 * SR))
    again = rand_result(rng, rng.randrange(0, 60), (stop - start) / SR, rng.random() < 0.2, rng.random() < 0.2)
    taken = [(a, a + rng.randrange(0, SR)) for a in sorted(rng.randrange(origin, origin + int(span * SR)) for _ in range(rng.randrange(0, 4)))]
    try:
        out, added = routes._merged(res, origin, again, start, stop, taken)
        info = routes._extract(out)
        assert len(info["tokens"]) == len(info["timestamps"])
        # the piece's own tokens all stay, in order
        it = iter(info["tokens"])
        base = routes._extract(res)["tokens"]
        assert all(any(t == x for x in it) for t in base), "piece tokens lost"
        routes._kept_near(res, origin, (origin, origin + int(span * SR)), origin + int(span * SR / 2))
    except Exception as exc:
        errors += 1
        if errors <= 5:
            print("case", case, repr(exc)); traceback.print_exc(limit=3)
print("errors", errors)
