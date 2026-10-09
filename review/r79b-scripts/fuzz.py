"""Fuzz: random speech, random skip in one piece, grid + jitter timing.

python fuzz.py <worktree> <seed> <n> [jitter]
Prints per-case outcome as JSON lines: case id, missing, dup, extra, inversions."""
from __future__ import annotations

import json
import os
import logging
import random
import sys

import sim

logging.disable(logging.CRITICAL)

seed, n = int(sys.argv[2]), int(sys.argv[3])
jitter = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
mode = sys.argv[5] if len(sys.argv) > 5 else "two"

for case in range(n):
    if os.environ.get("CASE") and int(os.environ["CASE"]) != case:
        continue
    rng = random.Random(seed * 100003 + case)
    if mode in ("two", "cut"):
        total = 120.0
        cut = rng.uniform(50, 62)
        ctx = 5.0
        ranges = [(0.0, cut), (cut, total)]
        windows = [(0.0, cut + ctx), (cut - ctx, total)]
        speech = [(0.0, total)]
    elif mode == "three":
        total = 180.0
        c1, c2 = rng.uniform(50, 62), rng.uniform(110, 122)
        ctx = 5.0
        ranges = [(0.0, c1), (c1, c2), (c2, total)]
        windows = [(0.0, c1 + ctx), (c1 - ctx, c2 + ctx), (c2 - ctx, total)]
        speech = [(0.0, total)]
    else:  # one piece
        total = rng.uniform(30, 75)
        ranges = [(0.0, total)]
        windows = [(0.0, total)]
        speech = None
    # words: a word every 0.25-0.7 s
    truth, t, i = [], rng.uniform(0.0, 0.5), 0
    while t < total - 0.2:
        truth.append((f"w{i}", round(t, 4)))
        i += 1
        t += rng.uniform(0.25, 0.7)
    piece = rng.randrange(len(ranges))
    ws, we = windows[piece]
    kind = rng.choice(["start", "mid", "end", "two"]) if mode != "cut" else rng.choice(["cutlow", "cuthigh"])
    if kind == "cutlow":  # piece 1 skips from its window start past the cut
        piece = 1
        a, b = windows[1][0], ranges[1][0] + rng.uniform(3.5, 15)
        skips = [(a, b)]
    elif kind == "cuthigh":  # piece 0 skips from before the cut to its window end
        piece = 0
        a, b = ranges[0][1] - rng.uniform(3.5, 15), windows[0][1]
        skips = [(a, b)]
    elif kind == "start":
        a, b = ws, rng.uniform(ws + 4, min(we, ws + 30))
        skips = [(a, b)]
    elif kind == "end":
        a = rng.uniform(max(ws, we - 30), we - 4)
        skips = [(a, we)]
    elif kind == "mid":
        a = rng.uniform(ws, we - 5)
        b = rng.uniform(a + 4, min(we, a + 30))
        skips = [(a, b)]
    else:
        a = rng.uniform(ws, ws + (we - ws) / 2 - 5)
        b = a + rng.uniform(4, 8)
        c = rng.uniform(b + 0.5, b + 5)
        d = min(we, c + rng.uniform(4, 8))
        skips = [(a, b), (c, d)]
    per_piece = [skips if p == piece else [] for p in range(len(ranges))]

    def jit(word, ws, _rng=rng):
        if jitter and random.Random(hash((word, round(ws, 3), seed))).random() < jitter:
            return random.Random(hash((word, ws, 1))).choice([-1, 1])
        return 0

    try:
        text, segments, words, calls = sim.run(
            truth, ranges, windows, speech, total, per_piece,
            redo_kw=dict({"jit": jit} if jitter else {}, **({"made_up_end": True} if os.environ.get("MADE_UP") else {})),
            first_kw={"jit": jit} if jitter else None,
        )
        s = sim.score(truth, words)
        if os.environ.get("CASE"):
            print("ranges", ranges, "windows", windows)
            lo, hi = float(os.environ["LO"]), float(os.environ["HI"])
            print("truth", [(w, t) for w, t in truth if lo <= t <= hi])
            print("words", [(w["word"], round(w["start"], 3)) for w in words if lo <= w["start"] <= hi])
        textwords = text.split()
        s["text_words_mismatch"] = textwords != [w["word"] for w in words]
    except Exception as exc:  # noqa: BLE001
        s = {"crash": repr(exc)}
        calls = []
    times = dict(truth)
    s["missing_t"] = [times[w] for w in s.get("missing", [])]
    print(json.dumps({"case": case, "kind": kind, "piece": piece, "skips": [[round(x, 2) for x in k] for k in skips],
                      "calls": calls, **s}))
