"""Spliced words near a piece's range edge (#73 seam): no duplicates or drops."""
import sys

sys.path.insert(0, sys.argv[1])
sys.path.insert(1, sys.argv[2])
from simlib import Model, run, words_every, word_list  # noqa: E402


def report(name, truth, out):
    want = [w for w, _t in word_list(truth)]
    got = [w["word"] for w in out.words]
    missing = [w for w in want if w not in got]
    dup = sorted({w for w in got if got.count(w) > 1})
    extra = [w for w in got if w not in want]
    starts = [w["start"] for w in out.words]
    mono = all(a <= b + 1e-9 for a, b in zip(starts, starts[1:]))
    print(f"{name}: words={len(got)}/{len(want)} missing={missing} dup={dup} extra={extra} "
          f"monotonic={mono} text==words={out.text == ' '.join(got)} calls={out.calls}")


cut = 60.04
ranges, windows = [(0, cut), (cut, 120)], [(0, cut + 5), (cut - 5, 120)]
speech = [(0, 120)]
first = {(0, cut + 5), (cut - 5, 120)}

# G1: piece 2 skips the start of its range; a word 30 ms after the cut; the redo times a frame early
truth = sorted(words_every(0.4, 59.9, 0.5) + [([" X"], [60.07])] + words_every(60.6, 120, 0.5, prefix="v"),
               key=lambda w: w[1][0])
early = lambda a, b: -0.08 if (round(a, 2), round(b, 2)) not in {(0, round(cut + 5, 2)), (round(cut - 5, 2), 120)} else 0.0
m = Model(truth, skips=lambda a, b: [(cut - 5, 70.0)] if (round(a, 2), b) == (round(cut - 5, 2), 120) else [], bias=early)
report("G1 skip at range start, word 30ms after cut, redo a frame early", truth, run(120, ranges, windows, speech, m))

# G2: same, no bias
m = Model(truth, skips=lambda a, b: [(cut - 5, 70.0)] if (round(a, 2), b) == (round(cut - 5, 2), 120) else [])
report("G2 same, no bias", truth, run(120, ranges, windows, speech, m))

# G3: piece 1 stops at 50 through its end; continuous words across the cut, redo a frame early
truth = words_every(0.2, 120, 0.5)
m = Model(truth, skips=lambda a, b: [(50.0, 70.0)] if (a, round(b, 2)) == (0, round(cut + 5, 2)) else [], bias=early)
report("G3 stop through range end, redo a frame early", truth, run(120, ranges, windows, speech, m))

# G4: piece 1 skips 50-60.04 but resumes in its right context; a word at the cut
truth = sorted(words_every(0.4, 59.9, 0.5) + [([" X"], [60.06])] + words_every(60.6, 120, 0.5, prefix="v"),
               key=lambda w: w[1][0])
m = Model(truth, skips=lambda a, b: [(50.0, 60.0)] if (a, round(b, 2)) == (0, round(cut + 5, 2)) else [], bias=early)
report("G4 skip to the cut, resumes just after it, redo a frame early", truth, run(120, ranges, windows, speech, m))
