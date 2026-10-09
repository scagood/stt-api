"""Stretch-edge words: a piece stops partway (mid-phrase), or skips a stretch
mid-way, in continuous speech. Every other window hears all. Compare to truth."""
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
    order_ok = [got.index(w) for w in want if w in got] == sorted(got.index(w) for w in want if w in got)
    starts = [w["start"] for w in out.words]
    mono = all(a <= b + 1e-9 for a, b in zip(starts, starts[1:]))
    text_ok = out.text == " ".join(got)
    print(f"{name}: words={len(got)}/{len(want)} missing={missing} dup={dup} extra={extra} "
          f"order_ok={order_ok} monotonic={mono} text==words={text_ok} calls={out.calls}")


ranges, windows = [(0, 60), (60, 120)], [(0, 65), (55, 120)]
speech = [(0, 120)]

# A: stops partway at 30 s (mid-phrase), 2-token words every 0.45 s
truth = words_every(0.2, 120, 0.45, tokens_per_word=2)
m = Model(truth, skips=lambda a, b: [(30.0, 65.0)] if (a, b) == (0, 65) else [])
report("A stop mid-phrase 0.45s/2tok", truth, run(120, ranges, windows, speech, m))

# B: skips 30-45 s mid-way in piece 1, one-token words every 0.3 s
truth = words_every(0.2, 120, 0.3)
m = Model(truth, skips=lambda a, b: [(30.0, 45.0)] if (a, b) == (0, 65) else [])
report("B skip 30-45 mid 0.3s/1tok", truth, run(120, ranges, windows, speech, m))

# C: same as B with 0.45s 2-token words
truth = words_every(0.2, 120, 0.45, tokens_per_word=2)
m = Model(truth, skips=lambda a, b: [(30.0, 45.0)] if (a, b) == (0, 65) else [])
report("C skip 30-45 mid 0.45s/2tok", truth, run(120, ranges, windows, speech, m))

# D: 0.5 s single-token words (slow): edges clear the tail
truth = words_every(0.2, 120, 0.5)
m = Model(truth, skips=lambda a, b: [(30.0, 45.0)] if (a, b) == (0, 65) else [])
report("D skip 30-45 mid 0.5s/1tok", truth, run(120, ranges, windows, speech, m))

# E: #77 shape: piece 2 skips the start of its range (60-70), whole-range redo hears it
truth = words_every(0.2, 120, 0.45, tokens_per_word=2)
m = Model(truth, skips=lambda a, b: [(55.0, 70.0)] if (a, b) == (55, 120) else [])
report("E skip range start 60-70", truth, run(120, ranges, windows, speech, m))

# F: only the first decode AND the whole-range redo skip (#77 long-window); short windows hear
truth = words_every(0.2, 120, 0.45, tokens_per_word=2)
m = Model(truth, skips=lambda a, b: [(30.0, 45.0)] if b - a > 30 else [])
report("F long windows all skip", truth, run(120, ranges, windows, speech, m))
