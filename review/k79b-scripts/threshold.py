"""Two skipped stretches in one piece, each holding one real word.

Main judges the piece's redo by all its stretches together (2 new words: kept);
the PR judges each stretch alone (1 word each: both dropped).
Also: a sub-3 s skip next to a 3 s+ one in the same piece."""
from __future__ import annotations

import logging
import sys

import sim

logging.basicConfig(level=logging.WARNING, format="%(message)s")
case = sys.argv[2]
ranges, windows, total = [(0.0, 60.0), (60.0, 120.0)], [(0.0, 65.0), (55.0, 120.0)], 120.0
speech = [(0.0, total)]
if case == "threshold":
    # fluent speech except two slow 4 s stretches (20-24 s, 30-34 s) each with one word
    truth = [(f"a{i}", 0.5 * i) for i in range(1, 40)]          # .. 19.5
    truth += [("Yes", 22.0)]                                      # 20-24: one word
    truth += [(f"b{i}", 24.0 + 0.5 * i) for i in range(12)]       # 24 .. 29.5
    truth += [("Right", 32.0)]                                    # 30-34: one word
    truth += [(f"c{i}", 34.0 + 0.5 * i) for i in range(172)]      # 34 .. 119.5
    skips = [[(21.0, 23.0), (31.0, 33.0)], []]
    speech = [(0.0, total)]
elif case == "short_gap":
    # piece 0 skips 10-12.5 s (2.5 s, under _STALL_SEC) and 30-38 s
    truth = [(f"w{i}", 0.25 + 0.5 * i) for i in range(239)]
    skips = [[(10.0, 12.5), (30.0, 38.0)], []]
else:
    raise SystemExit(case)
text, segments, words, calls = sim.run(sorted(truth, key=lambda x: x[1]), ranges, windows, speech, total, skips)
print("redo calls:", calls)
print("score:", sim.score(truth, words))
