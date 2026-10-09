"""The earlier review's repros, with main's whole-range redo skipping the same
stretch again (so main drops it and only the stretch redo can recover it):
any decode of 40 s or more skips the stretch; shorter ones hear it.

python stretch_repros.py <worktree>"""
import asyncio
import logging
import sys

sys.path.insert(0, sys.argv[1])
sys.path.insert(1, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/k79a-scripts")
import simlib  # noqa: E402
from simlib import Model, run, words_every, word_list, S  # noqa: E402
from parakeet_service import routes  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="   log: %(message)s")


def report(name, truth, out, keys=()):
    want = [w for w, _t in word_list(truth)]
    got = [w["word"] for w in out.words]
    missing = [w for w in want if w not in got]
    dup = sorted({w for w in got if got.count(w) > 1})
    extra = [w for w in got if w not in want]
    pos = [got.index(w) for w in want if w in got]
    order_ok = pos == sorted(pos)
    starts = [w["start"] for w in out.words]
    mono = all(a <= b + 1e-9 for a, b in zip(starts, starts[1:]))
    text_ok = out.text == " ".join(got)
    seg_ok = out.text == routes._clean_text(" ".join(s["segment"] for s in out.segments))
    print(f"{name}: words={len(got)}/{len(want)} missing={missing[:12]}{'...' if len(missing) > 12 else ''} "
          f"dup={dup} extra={extra} order_ok={order_ok} monotonic={mono} text==words={text_ok} text==segs={seg_ok}")
    print(f"   redo calls={out.calls[len(out.prep.ranges):]}")
    for k in keys:
        print(f"   {k!r}: {got.count(k)} in words, {out.text.split().count(k)} in text, "
              f"at {[round(w['start'], 2) for w in out.words if w['word'] == k]}")


ranges, windows = [(0, 60), (60, 120)], [(0, 65), (55, 120)]
speech = [(0, 120)]
LONG = 40


def long_skips(stretches):
    return lambda a, b: stretches if b - a >= LONG else []


# 1. edge words: "we were under|stand|ing" .. "the" at 20.08 .. "of" at 27.76 before "Rome" at 28.0
truth = [([f" w{i}"], [0.5 * i]) for i in range(1, 39)]
truth += [([" we"], [19.2]), ([" were"], [19.36]), ([" under", "stand", "ing"], [19.52, 19.68, 19.84])]
truth += [([" the"], [20.08]), ([" history"], [20.4])]
truth += [([f" s{i}"], [21.0 + 0.5 * i]) for i in range(13)]
truth += [([" part"], [27.4]), ([" of"], [27.76]), ([" Rome", "."], [28.0, 28.3])]
truth += [([f" v{i}"], [28.8 + 0.5 * i]) for i in range(183)]
m = Model(truth, skips=long_skips([(20.0, 27.99)]))
report("1 edges (the 20.08, of 27.76, Rome 28.0)", truth, run(120, ranges, windows, speech, m), keys=("the", "of", "Rome."))

# 2. stop mid-phrase at 30 s, 2-token words every 0.45 s
truth = words_every(0.2, 120, 0.45, tokens_per_word=2)
m = Model(truth, skips=lambda a, b: [(30.0, 65.0)] if b - a >= LONG and a < 30 else [])
report("2 stop at 30 s, 0.45 s 2-token words", truth, run(120, ranges, windows, speech, m))

# 3. skip 30-45 s, words every 0.3 s
truth = words_every(0.2, 120, 0.3)
m = Model(truth, skips=long_skips([(30.0, 45.0)]))
report("3 skip 30-45 s, 0.3 s words", truth, run(120, ranges, windows, speech, m))

# 4. one stray word at 32.6 s inside a 30-45 s skip (0.5 s words)
truth = words_every(0.2, 120, 0.5)
m = Model(truth, skips=long_skips([(30.0, 32.6), (32.8, 45.0)]))
report("4 stray word at 32.6 in 30-45 s skip", truth, run(120, ranges, windows, speech, m))

# 5. a piece skipping 10-12.5 s and 30-38 s
truth = [([f" w{i}"], [0.25 + 0.5 * i]) for i in range(239)]
m = Model(truth, skips=long_skips([(10.0, 12.5), (30.0, 38.0)]))
report("5 skip 10-12.5 and 30-38 s", truth, run(120, ranges, windows, speech, m))

# 6. two 4 s stretches each with one word ("Yes", "Right")
truth = [([f" a{i}"], [0.5 * i]) for i in range(1, 40)] + [([" Yes"], [22.0])]
truth += [([f" b{i}"], [24.0 + 0.5 * i]) for i in range(12)] + [([" Right"], [32.0])]
truth += [([f" c{i}"], [34.0 + 0.5 * i]) for i in range(172)]
m = Model(truth, skips=long_skips([(21.0, 23.0), (31.0, 33.0)]))
out = run(120, ranges, windows, speech, m)
report("6 Yes/Right (VAD speech throughout)", truth, out, keys=("Yes", "Right"))
# with VAD calling 20-24 and 30-34 speech but quiet around them is not needed: speech throughout is the worst case

# 7. cut-straddling X at 60.07 / 60.06, cut at 60.04, range redo skips again, redo a frame early
cut = 60.04
r2, w2 = [(0, cut), (cut, 120)], [(0, cut + 5), (cut - 5, 120)]
early = lambda a, b: -0.08 if b - a < LONG else 0.0  # noqa: E731
truth = sorted(words_every(0.4, 59.9, 0.5) + [([" X"], [60.07])] + words_every(60.6, 120, 0.5, prefix="v"), key=lambda w: w[1][0])
m = Model(truth, skips=lambda a, b: [(cut - 5, 70.0)] if b - a >= LONG and a > 50 else [], bias=early)
report("7a X at 60.07, next piece skips its start, redo a frame early", truth, run(120, r2, w2, speech, m), keys=("X",))
m = Model(truth, skips=lambda a, b: [(cut - 5, 70.0)] if b - a >= LONG and a > 50 else [])
report("7b same, no bias", truth, run(120, r2, w2, speech, m), keys=("X",))
truth = sorted(words_every(0.4, 59.9, 0.5) + [([" X"], [60.06])] + words_every(60.6, 120, 0.5, prefix="v"), key=lambda w: w[1][0])
m = Model(truth, skips=lambda a, b: [(50.0, 60.0)] if b - a >= LONG and a < 10 else [], bias=early)
report("7c skip to the cut, X at 60.06 just past it, redo a frame early", truth, run(120, r2, w2, speech, m), keys=("X",))
m = Model(truth, skips=lambda a, b: [(50.0, 60.0)] if b - a >= LONG and a < 10 else [], bias=lambda a, b: 0.08 if b - a < LONG else 0.0)
report("7d same, redo a frame late", truth, run(120, r2, w2, speech, m), keys=("X",))
