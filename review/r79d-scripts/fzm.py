"""Random-input fuzz of the head's _merged, _heard_near, _redo_windows, _untimed: any exception,
lost piece token, or out-of-order output times (when inputs were sorted)?"""
import random
import sys
import traceback
from types import SimpleNamespace

import lib
from lib import routes

SR = 16000
TOK = [" the", "▁the", " ", "▁", "ing", ",", ".", " 1", "9", "8", "4", " £", "'s", " ", "-", "?", " a", " no",
       " I", "Ā", " 中", "文", "", " .", "...", " -", " the", " the", " no", " a"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30000


def rand_result(rng, n, span, unsorted=False, bad=False):
    toks = [rng.choice(TOK) for _ in range(n)]
    ts = sorted(round(rng.uniform(0, span) / 0.08) * 0.08 for _ in range(n))
    if unsorted:
        rng.shuffle(ts)
    if bad:
        for k in range(n):
            r = rng.random()
            if r < 0.05:
                ts[k] = float("nan")
            elif r < 0.1:
                ts[k] = -1.0
            elif r < 0.12:
                ts[k] = None
            elif r < 0.14:
                ts[k] = float("inf")
        if rng.random() < 0.2:
            ts = ts[: rng.randrange(n + 1)]
    return SimpleNamespace(text="".join(toks), tokens=toks, timestamps=ts)


errors = unsorted_out = lost = 0
for case in range(N):
    rng = random.Random(case)
    span = rng.uniform(1, 80)
    origin = rng.randrange(0, 10 * SR)
    uns, bad = rng.random() < 0.2, rng.random() < 0.2
    res = rand_result(rng, rng.randrange(0, 80), span, uns, bad)
    own = (origin + rng.choice([0, rng.randrange(0, 5 * SR)]), origin + int(span * SR) - rng.choice([0, rng.randrange(0, 5 * SR)]))
    start = origin + rng.randrange(0, int(span * SR))
    stop = start + rng.randrange(1, int(20 * SR))
    uns2, bad2 = rng.random() < 0.2, rng.random() < 0.2
    again = rand_result(rng, rng.randrange(0, 80), (stop - start) / SR, uns2, bad2)
    stretches = []
    for _ in range(rng.randrange(0, 4)):
        a = rng.randrange(start, stop)
        b = rng.randrange(a, stop + 1)
        stretches.append((a, b))
    stretches.sort()
    if rng.random() < 0.3:
        stretches = [(own[0], min(own[1], own[0] + 3 * SR))]
    if rng.random() < 0.3:
        stretches = [(max(own[0], own[1] - 3 * SR), own[1])]
    taken = []
    for a in sorted(rng.randrange(origin, origin + int(span * SR)) for _ in range(rng.randrange(0, 6))):
        taken.append((a, a + rng.randrange(0, SR), rng.choice(["the", "a", "", "no", "x"])))
    try:
        out, added = routes._merged(res, origin, again, start, stop, stretches, own, taken)
        info = routes._extract(out)
        assert len(info["tokens"]) == len(info["timestamps"])
        it = iter(info["tokens"])
        base = routes._extract(res)["tokens"]
        if not all(any(t == x for x in it) for t in base):
            lost += 1
        if not (uns or bad or uns2 or bad2):
            ts = info["timestamps"]
            if any(b < a - 1e-9 for a, b in zip(ts, ts[1:])):
                unsorted_out += 1
        routes._heard_near(res, origin, origin + int(span * SR / 2))
        routes._redo_windows(own, stretches)
    except Exception as exc:  # noqa: BLE001
        errors += 1
        if errors <= 5:
            print("case", case, repr(exc))
            traceback.print_exc(limit=4)
print("cases", N, "errors", errors, "piece tokens lost", lost, "unsorted output times (sorted inputs)", unsorted_out)
