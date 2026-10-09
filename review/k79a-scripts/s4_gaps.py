"""Words the whole-piece redo (main) recovers that a stretch-only redo (PR)
cannot: a stray word splits the skip, leaving a part under 3 s; and VAD calls."""
import sys

sys.path.insert(0, sys.argv[1])
sys.path.insert(1, sys.argv[2])
from simlib import Model, run, words_every, word_list  # noqa: E402
from parakeet_service import routes  # noqa: E402


def report(name, truth, out):
    want = [w for w, _t in word_list(truth)]
    got = [w["word"] for w in out.words]
    missing = [w for w in want if w not in got]
    dup = sorted({w for w in got if got.count(w) > 1})
    print(f"{name}: words={len(got)}/{len(want)} missing={missing} dup={dup} calls={out.calls}")


ranges, windows = [(0, 60), (60, 120)], [(0, 65), (55, 120)]
speech = [(0, 120)]

# I: skip 30-45 s, but the first decode hears one stray word at 32.5 s in it (slow, 0.5 s words)
truth = words_every(0.2, 120, 0.5)
stray = [(30.0, 32.6), (32.8, 45.0)]
m = Model(truth, skips=lambda a, b: stray if (a, b) == (0, 65) else [])
report("I stray word splits the skip 2.3 s + 12 s", truth, run(120, ranges, windows, speech, m))

# J: one-piece 40 s clip, words every 1 s, no stretch: VAD calls
calls = []
if hasattr(routes, "speech_segments"):
    real = routes.speech_segments
truth = words_every(0.5, 40, 1.0)
m = Model(truth)
import simlib  # noqa: E402
orig_run = simlib.run


def counting_vad(wav):
    calls.append(wav.size)
    return [(0, wav.size)]


if hasattr(routes, "speech_segments"):
    routes.speech_segments = counting_vad
out = simlib.prepared  # noqa
import asyncio  # noqa: E402
prep = simlib.prepared(40, [(0, 40)], [(0, 40)], None)
worker = simlib.Worker(m)
res = asyncio.run(worker.submit_many(prep.pieces, "x"))
again = asyncio.run(routes._redo_stalled(simlib.request(worker), [prep], res, "parakeet-v3:fp32"))
print(f"J no stretch: VAD calls={len(calls)} same results={again == res} decodes={m.calls}")

# K: one-piece 40 s, a 3.5 s pause (silence) mid-way: VAD runs once, nothing redone
calls.clear()
truth = words_every(0.5, 20, 1.0) + words_every(23.8, 40, 1.0, prefix="v")
m = Model(truth)
prep = simlib.prepared(40, [(0, 40)], [(0, 40)], None)
if hasattr(routes, "speech_segments"):
    routes.speech_segments = lambda wav: calls.append(wav.size) or [(0, 19.9 * 16000), (23.7 * 16000, wav.size)]
worker = simlib.Worker(m)
res = asyncio.run(worker.submit_many(prep.pieces, "x"))
again = asyncio.run(routes._redo_stalled(simlib.request(worker), [prep], res, "parakeet-v3:fp32"))
print(f"K pause: VAD calls={len(calls)} same results={again == res} decodes={m.calls}")
