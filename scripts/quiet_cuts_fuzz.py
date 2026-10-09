#!/usr/bin/env python3
"""Where forced cuts land in speech with no pause, against main's even split.

Random layouts of phrases, some longer than a chunk, built of words with
gaps, syllable dips and stop closures in them, at several gap depths. VAD is
replaced by the layout's own phrases, so only the cut placement is measured.
For each bucket and chunk-bound configuration it prints, for main (the even
split) and for chunker._quiet_cuts:

- pieces, and forced cuts (inside a phrase)
- forced cuts inside a word, and how many layouts have more of them than main
- ranges under 2 s, speech outside every range, bound violations
- decoded audio main's windows cover that these don't (margin given up)

    python scripts/quiet_cuts_fuzz.py [layouts per bucket] [bucket ...]
"""
from __future__ import annotations

import bisect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parakeet_service import chunker  # noqa: E402

SR = chunker.TARGET_SR
CONFIGS = {  # target, max, min, context; Whisper's 25/30 bounds are v2 ctx0's
    "v2 ctx5": (25, 30, 20, 5),
    "v2 ctx0": (25, 30, 20, 0),
    "v3 ctx5": (60, 75, 20, 5),
}
# gap depth in dB under the word (low, high), closures: (share of words, depth), word level spread in dB
BUCKETS = {
    "gaps 30 dB": dict(gap=(28, 32), closures=None, spread=4),
    "gaps 20 dB": dict(gap=(18, 22), closures=None, spread=4),
    "gaps 10 dB": dict(gap=(9, 11), closures=None, spread=4),
    "gaps 6 dB": dict(gap=(5, 7), closures=None, spread=4),
    "closures 40%, gaps 20 dB": dict(gap=(18, 22), closures=(0.4, (25, 35)), spread=4),
    "closures 50%, gaps 8-15 dB": dict(gap=(8, 15), closures=(0.5, (25, 35)), spread=4),
    "uneven words ±8 dB, gaps 10 dB": dict(gap=(9, 11), closures=None, spread=8),
}


def layout(rng, bucket, scale, seconds=240):
    """Audio, VAD phrases and word spans (samples) for one random layout."""
    total = int(seconds * SR)
    wav = (rng.standard_normal(total) * 10 ** (-60 / 20)).astype(np.float32)
    phrases, words, at = [], [], rng.uniform(0.2, 2)
    while True:
        length = (rng.uniform(15, 130) if rng.random() < 0.3 else rng.uniform(1, 8)) * scale
        if at + length > seconds - 3.5:
            break
        level = rng.uniform(-30, -12)
        t, end = at, at + length
        while t < end:
            w = min(rng.uniform(0.15, 0.5), end - t)
            a, b = int(t * SR), int((t + w) * SR)
            db = level + rng.uniform(-bucket["spread"], bucket["spread"])
            env = np.ones(b - a, dtype=np.float32)
            for _ in range(int(w > 0.25) + 1):  # syllable dips 6-10 dB, 40-80 ms
                d = int(rng.uniform(0.04, 0.08) * SR)
                if b - a > d + 2:
                    s = rng.integers(0, b - a - d)
                    env[s: s + d] = 10 ** (-rng.uniform(6, 10) / 20)
            if bucket["closures"] and rng.random() < bucket["closures"][0] and w > 0.18:
                d = int(rng.uniform(0.06, 0.12) * SR)  # a stop closure: p, t, k
                if b - a > d + int(0.06 * SR):
                    s = rng.integers(int(0.03 * SR), b - a - d - int(0.03 * SR))
                    env[s: s + d] = 10 ** (-rng.uniform(*bucket["closures"][1]) / 20)
            wav[a:b] = rng.standard_normal(b - a).astype(np.float32) * 10 ** (db / 20) * env
            words.append((a, b))
            t += w
            if t >= end:
                break
            g = min(rng.uniform(0.05, 0.35), end - t)
            a, b = int(t * SR), int((t + g) * SR)
            wav[a:b] = rng.standard_normal(b - a).astype(np.float32) * 10 ** ((level - rng.uniform(*bucket["gap"])) / 20)
            t += g
        phrases.append((int(at * SR), int(end * SR)))
        r = rng.random()
        at = end + (rng.uniform(0.4, 1.5) if r < 0.7 else rng.uniform(1.5, 3) if r < 0.9 else rng.uniform(3, 8))
    return wav, phrases, words


def _inside(point, spans, starts):
    index = bisect.bisect_right(starts, point) - 1
    return index >= 0 and spans[index][0] < point < spans[index][1]


def _covered(spans):
    """Total samples in the union of `spans`."""
    total, reach = 0, -1
    for a, b in sorted(spans):
        if b > reach:
            total += b - max(a, reach)
            reach = b
    return total


def measure(wav, phrases, words, bounds):
    target, maximum, minimum, context = bounds
    chunker._speech_segments = lambda _wav: phrases
    plan = chunker.plan_chunks(wav, target_sec=target, max_sec=maximum, min_sec=minimum, context_sec=context)
    ranges, windows = plan.ranges, plan.windows
    own = int((maximum - 2 * context) * SR)
    phrase_starts, word_starts = [s for s, _ in phrases], [s for s, _ in words]
    cuts = [a for (_, a), (b, _) in zip(ranges, ranges[1:]) if a == b]
    forced = [c for c in cuts if _inside(c, phrases, phrase_starts)]
    lost = sum(e - s for s, e in phrases) - sum(
        max(0, min(e, b) - max(s, a)) for s, e in phrases for a, b in ranges)
    bad = sum(b - a > own for a, b in ranges)
    bad += sum(wb - wa > int(maximum * SR) or not wa <= a < b <= wb for (a, b), (wa, wb) in zip(ranges, windows))
    bad += sum(left[1] > right[0] for left, right in zip(ranges, ranges[1:]))
    return dict(
        pieces=len(ranges), forced=len(forced), in_word=sum(_inside(c, words, word_starts) for c in forced),
        short=sum(b - a < 2 * SR for a, b in ranges), lost=lost, bad=bad,
    ), windows


def run(layouts, buckets, seed=0):
    quiet = chunker._quiet_cuts
    for name in buckets:
        for scale in (1.0, 1.6):
            rng = np.random.default_rng(seed)
            totals = {config: [{}, {}, 0, 0] for config in CONFIGS}  # main, PR, layouts worse, margin samples given up
            for _ in range(layouts):
                wav, phrases, words = layout(rng, BUCKETS[name], scale)
                for config, bounds in CONFIGS.items():
                    chunker._quiet_cuts = lambda _wav, even, *_edges: even
                    before, main_windows = measure(wav, phrases, words, bounds)
                    chunker._quiet_cuts = quiet
                    after, windows = measure(wav, phrases, words, bounds)
                    row = totals[config]
                    for key in before:
                        row[0][key] = row[0].get(key, 0) + before[key]
                        row[1][key] = row[1].get(key, 0) + after[key]
                    row[2] += after["in_word"] > before["in_word"] or after["pieces"] > before["pieces"]
                    row[3] += _covered(main_windows + windows) - _covered(windows)
            for config, (m, p, worse, given_up) in totals.items():
                share = lambda row: f"{row['in_word']} ({row['in_word'] / max(1, row['forced']):.1%})"
                print(
                    f"{name:32} x{scale} {config}: pieces {m['pieces']}->{p['pieces']}, forced {m['forced']}->{p['forced']}, "
                    f"in a word {share(m)} -> {share(p)}, layouts worse {worse}, <2 s {m['short']}->{p['short']}, "
                    f"speech lost {m['lost']}->{p['lost']}, bounds broken {m['bad']}->{p['bad']}, "
                    f"margin given up {given_up / SR:.1f} s",
                    flush=True,
                )


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    run(count, sys.argv[2:] or list(BUCKETS))
