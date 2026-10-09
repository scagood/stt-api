#!/usr/bin/env python3
"""Forced cuts inside words on real audio: main's even split against
chunker._quiet_cuts, by each model's chunk bounds.

    PARAKEET_VAD_GATE_DB=-45 python scripts/quiet_cuts_real.py AUDIO.wav WORDS.json [...]

AUDIO.wav is 16 kHz mono; WORDS.json is the service's verbose_json for the
same audio (or a cleaner copy of it), ideally with an aligner's word times
(aligner=wav2vec2-base-960h). The README's figures came from LibriVox
chapters under pink noise at -40 and -35 dBFS, with the clean chapter's
words, so that a fixed gate below the noise hears each as one stretch.
A cut counts as deep where it is over 40 ms from both edges of its word.
"""
from __future__ import annotations

import bisect
import json
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parakeet_service import chunker  # noqa: E402

SR = chunker.TARGET_SR
BOUNDS = {  # target, max, min, context
    "parakeet-v2 (20 s)": (25, 30, 20, 5),
    "30 s (Whisper, no context)": (25, 30, 20, 0),
    "parakeet-v3 (65 s)": (60, 75, 20, 5),
}


def forced_in_words(wav, words, bounds):
    target, maximum, minimum, context = bounds
    plan = chunker.plan_chunks(wav, target_sec=target, max_sec=maximum, min_sec=minimum, context_sec=context)
    starts = [start for start, _end in words]
    inside = deep = forced = 0
    for (_start, cut), (following, _end) in zip(plan.ranges, plan.ranges[1:]):
        if cut != following or not any(a < cut < b for a, b in plan.speech):
            continue
        forced += 1
        at = cut / SR
        index = bisect.bisect_right(starts, at) - 1
        if index >= 0 and words[index][0] < at < words[index][1]:
            inside += 1
            deep += min(at - words[index][0], words[index][1] - at) > 0.04
    return len(plan.ranges), forced, inside, deep


def main(pairs):
    quiet = chunker._quiet_cuts
    for audio, transcript in zip(pairs[::2], pairs[1::2]):
        with wave.open(audio) as handle:
            wav = np.frombuffer(handle.readframes(handle.getnframes()), np.int16).astype(np.float32) / 32768
        words = [(w["start"], w["end"]) for w in json.load(open(transcript))["words"]]
        for name, bounds in BOUNDS.items():
            chunker._quiet_cuts = lambda _wav, even, *_edges: even
            before = forced_in_words(wav, words, bounds)
            chunker._quiet_cuts = quiet
            after = forced_in_words(wav, words, bounds)
            print(
                f"{Path(audio).name} {name}: pieces {before[0]}->{after[0]}, forced cuts {before[1]}->{after[1]}, "
                f"in a word {before[2]}->{after[2]}, over 40 ms in {before[3]}->{after[3]}",
                flush=True,
            )


if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) % 2 == 0:
        sys.exit(__doc__)
    main(sys.argv[1:])
