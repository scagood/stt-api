"""Deterministic repro: a piece skips a stretch of fluent speech; the redo hears
it all. Which words end up in the transcript, on main vs the PR?

python edge.py <worktree> <case>"""
from __future__ import annotations

import logging
import sys

import sim

logging.basicConfig(level=logging.WARNING, format="%(message)s")
case = sys.argv[2]

# Fluent speech, 0.3 s between word starts ("of the", "in a" ...) where marked.
if case == "end_edge":
    # two pieces cut at 60 s, 5 s context; piece 0 skips 20.0-28.0 s.
    # Words every 0.5 s, except "the" at 27.9 s, 0.2 s before "cat" at 28.1 s.
    truth = [(f"w{i}", 0.5 * i) for i in range(1, 56)]  # 0.5 .. 27.5
    truth += [("the", 27.9), ("cat", 28.1)] + [(f"v{i}", 28.6 + 0.5 * i) for i in range(70)]
    truth.sort(key=lambda x: x[1])
    ranges, windows = [(0.0, 60.0), (60.0, 120.0)], [(0.0, 65.0), (55.0, 120.0)]
    skips = [[(20.0, 28.0)], []]
    total = 120.0
    speech = [(0.0, total)]
elif case == "start_edge":
    # piece 0 hears "sat" at 19.9 s, then skips: "on" at 20.15 s (0.25 s later) is lost
    truth = [(f"w{i}", 0.5 * i) for i in range(1, 40)]  # .. 19.5
    truth += [("sat", 19.9), ("on", 20.15)] + [(f"v{i}", 20.6 + 0.5 * i) for i in range(198)]
    truth.sort(key=lambda x: x[1])
    ranges, windows = [(0.0, 60.0), (60.0, 120.0)], [(0.0, 65.0), (55.0, 120.0)]
    skips = [[(20.0, 28.0)], []]
    total = 120.0
    speech = [(0.0, total)]
elif case == "one_piece_end_edge":
    # #77's shape: a one-piece 70 s clip that skips its opening; the last word
    # of the opening, "the", is 0.24 s before "Part" where the piece resumes
    truth = [(f"o{i}", 0.5 + 0.5 * i) for i in range(19)]  # 0.5 .. 9.5
    truth += [("the", 10.16), ("Part", 10.4)] + [(f"w{i}", 11.0 + 0.5 * i) for i in range(115)]
    truth.sort(key=lambda x: x[1])
    ranges = windows = [(0.0, 70.0)]
    skips = [[(0.0, 10.3)]]
    total = 70.0
    speech = None
else:
    raise SystemExit(case)

text, segments, words, calls = sim.run(truth, ranges, windows, speech, total, skips)
print("redo calls:", calls)
print("score:", sim.score(truth, words))
names = [w["word"] for w in words]
for key in ("the", "on", "sat", "cat", "Part"):
    if any(key == w for w, _t in truth):
        print(f"{key!r} in words: {key in names}; in text: {key in text.split()}")
