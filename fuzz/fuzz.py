"""Fake-decode fuzz for #79: drive the real _redo_stalled and _stitch of
main, an earlier head and the code under test on the same fake decodes, and
count the words each loses, duplicates or invents against the truth and
against main. See README.md.

Usage: python -I fuzz/fuzz.py CASES KINDS MISHEAR [repeat=R split=S madeup=M drift=D offset=N] [-v]
"""
from __future__ import annotations

import asyncio
import importlib
import io
import logging
import os
import random
import re
import subprocess
import sys
import tarfile
import types
import zlib
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("FUZZ_REPO") or os.path.dirname(HERE)  # the checkout this folder is in
CACHE = os.path.join(HERE, ".versions")
BASE = os.environ.get("FUZZ_BASE", "ed4be7f")  # main as #79 forked it: what the others are measured against
OTHER = os.environ.get("FUZZ_OTHER", "e4deaa7")  # an earlier head, for comparison; empty to skip
HEAD = os.environ.get("FUZZ_HEAD", "worktree")  # the code under test: this checkout's, or a git rev
ort = types.ModuleType("onnxruntime")
ort.get_available_providers = lambda: ["CPUExecutionProvider"]
sys.modules.setdefault("onnx_asr", types.ModuleType("onnx_asr"))
sys.modules.setdefault("onnxruntime", ort)
logging.disable(logging.CRITICAL)


def _unpacked(rev, package):
    """routes of parakeet_service at `rev`, unpacked under CACHE as `package`."""
    target = os.path.join(CACHE, rev)
    if not os.path.isdir(os.path.join(target, package)):
        os.makedirs(target, exist_ok=True)
        archive = subprocess.run(["git", "-C", REPO, "archive", rev, "parakeet_service"], check=True, capture_output=True).stdout
        tarfile.open(fileobj=io.BytesIO(archive)).extractall(target, filter="data")
        os.rename(os.path.join(target, "parakeet_service"), os.path.join(target, package))
    sys.path.insert(0, target)
    return importlib.import_module(f"{package}.routes")


def _versions():
    versions = {"parakeet_main": _unpacked(BASE, "parakeet_main")}
    if OTHER:
        versions["parakeet_v2"] = _unpacked(OTHER, "parakeet_v2")
    if HEAD == "worktree":
        sys.path.insert(0, REPO)
        versions["parakeet_service"] = importlib.import_module("parakeet_service.routes")
    else:
        versions["parakeet_service"] = _unpacked(HEAD, "parakeet_head")
    return versions


SR = 16_000
FRAME = 0.08
REPEAT, SPLIT, MADEUP, DRIFT, OFFSET = 0.0, 0.0, 0.0, 0.0, 0


def spelling(i):
    """A word's spelling: a repeat (negative id) is spelt as the word it repeats."""
    return f"w{-i - 1}" if i < 0 else f"w{i}"


VERSIONS = _versions()  # reported as main, v2 (the earlier head) and service (under test)


def truth_words(rng, total, pauses):
    """(id, start, token offsets) for speech filling `total` seconds."""
    words, at, i = [], rng.uniform(0.0, 0.6), 0
    while at < total - 0.3:
        tokens = rng.choice([1, 1, 2, 2, 3])
        offsets = [0.0]
        for _ in range(tokens - 1):
            offsets.append(offsets[-1] + rng.choice([0.08, 0.08, 0.16, 0.24]))
        words.append((i, at, offsets))
        i += 1
        if rng.random() < REPEAT and words:
            at += offsets[-1] + rng.uniform(0.12, 0.3)
            words.append((-words[-1][0] - 1 if words[-1][0] >= 0 else words[-1][0], at, [0.0]))
        at += offsets[-1] + rng.uniform(0.12, 0.45)
        if pauses and rng.random() < 0.03:
            at += rng.uniform(1.0, 4.5)
    return words


class Model:
    """Deterministic per window: the same audio decodes the same."""

    def __init__(self, seed, words, wav, mishear):
        self.seed, self.words, self.wav, self.mishear = seed, words, wav, mishear

    def decode(self, a, b):
        rng = random.Random(zlib.crc32(f"{self.seed}:{a}:{b}".encode()))
        sa, sb = a / SR, b / SR
        skips = []
        for _ in range(rng.choice([0, 1, 1, 2])):
            start = rng.uniform(sa, sb)
            skips.append((start, start + rng.choice([rng.uniform(3.0, 12.0), rng.uniform(0.5, 3.0), 1e9])))
        tokens, times = [], []
        for i, start, offsets in self.words:
            if not sa <= start < sb or any(lo <= start < hi for lo, hi in skips):
                continue
            if start > sb - 0.2 and rng.random() < 0.5:
                continue  # cut off as the input ends
            rel = round((start - sa) / FRAME) * FRAME + rng.choice([-FRAME, 0, 0, 0, FRAME])
            if start > sb - 1.5 and rng.random() < DRIFT:
                rel -= rng.choice([0.32, 0.4, 0.48])  # timed early as the input ends
            rel = max(0.0, rel)
            name = spelling(i)
            if rng.random() < self.mishear:
                name = "v" + name[1:]
            if rng.random() < SPLIT:
                gap = rng.choice([0.16, 0.24, 0.32, 0.4])
                tokens += [f" {name}a", f" {name}b"]
                times += [round(rel, 2), round(rel + gap, 2)]
                continue
            for k, offset in enumerate(offsets):
                tokens.append(f" {name}" if k == 0 else "x")
                times.append(round(rel + offset, 2))
        if rng.random() < MADEUP and times:
            at = min(sb - sa - 0.05, times[-1] + rng.uniform(0.1, 0.9))
            tokens.append(f" u{a}")  # made up as the input ends (#68)
            times.append(round(max(times[-1], at), 2))
        return SimpleNamespace(text="".join(t.replace("▁", " ") for t in tokens).strip(), tokens=tokens, timestamps=times)


class Worker:
    def __init__(self, model):
        self.model, self.calls = model, 0

    async def submit_many(self, pieces, _key):
        base = self.model.wav.__array_interface__["data"][0]
        out = []
        for piece in pieces:
            self.calls += 1
            a = (piece.__array_interface__["data"][0] - base) // 4
            out.append(self.model.decode(a, a + piece.size))
        return out


def case(seed, kind, mishear):
    rng = random.Random(seed)
    if kind == "one":
        total = rng.uniform(20.0, 75.0)
        ranges = [(0.0, total)]
        windows = ranges
    else:
        count = {"two": 2, "three": 3}[kind]
        cuts = [0.0] + [60.0 * k + rng.uniform(-1.0, 1.0) for k in range(1, count)] + [60.0 * count]
        total = cuts[-1]
        ranges = list(zip(cuts, cuts[1:]))
        windows = [(max(0.0, s - 5.0) if k else s, min(total, e + 5.0) if k < count - 1 else e) for k, (s, e) in enumerate(ranges)]
    words = truth_words(rng, total, pauses=rng.random() < 0.3)
    wav = np.zeros(int(total * SR), dtype=np.float32)
    speech = [(0, wav.size)]
    if any(b - a > 3.0 for (_i, a, _o), (_j, b, _p) in zip(words, words[1:])):
        speech = []  # VAD silent in the long pauses
        start = max(0.0, words[0][1] - 0.12)
        for (_i, a, o), (_j, b, _p) in zip(words, words[1:]):
            if b - (a + o[-1]) > 0.6:
                speech.append((int(start * SR), int((a + o[-1] + 0.3) * SR)))
                start = b - 0.12
        speech.append((int(start * SR), wav.size))
    return SimpleNamespace(seed=seed, words=words, wav=wav, ranges=ranges, windows=windows, speech=speech, mishear=mishear, one=kind == "one")


class Said(list):
    half = None


def run(routes, c):
    samples = lambda spans: [(int(a * SR), int(b * SR)) for a, b in spans]
    ranges, windows = samples(c.ranges), samples(c.windows)
    ranges[-1] = (ranges[-1][0], c.wav.size)
    windows[-1] = (windows[-1][0], c.wav.size)
    if c.one:
        ranges = windows = [(0, c.wav.size)]
    prepared = routes._PreparedAudio(
        waveform=c.wav, ranges=ranges, windows=windows,
        speech=(None if c.one and hasattr(routes, "speech_segments") else ([] if c.one else c.speech)),
        pieces=[c.wav[a:b] for a, b in windows], duration=c.wav.size / SR,
    )
    if c.one and hasattr(routes, "speech_segments"):
        routes.speech_segments = lambda wav, speech=c.speech: speech
    model = Model(c.seed, c.words, c.wav, c.mishear)
    worker = Worker(model)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(worker=worker, ready=True, audio_pool=None)))

    async def go():
        results = await routes._infer(request, prepared.pieces, "parakeet-v3:int8")
        results = await routes._redo_stalled(request, [prepared], results, "parakeet-v3:int8")
        return routes._stitch(prepared, results)

    text, _segments, words = asyncio.run(go())
    said = Said()
    for word in text.split():
        m = re.fullmatch(r"([wvsu])(\d+)x*([ab]?)", word.strip(".,"))
        if not m:
            said.append(word)
        elif m.group(1) == "u":
            said.append(word)  # made up
        elif m.group(3) == "b" and said and said[-1] == f"w{m.group(2)}" and getattr(said, "half", None) == len(said) - 1:
            continue  # the second half of a split reading after its first
        else:
            said.append(f"w{m.group(2)}")
            if m.group(3) == "a":
                said.half = len(said) - 1
    return said, worker.calls - len(prepared.pieces)


def main():
    global REPEAT, SPLIT, MADEUP, DRIFT, OFFSET
    from collections import Counter
    cases = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    kinds = (sys.argv[2] if len(sys.argv) > 2 else "one,two,three").split(",")
    mishear = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    for arg in sys.argv[4:]:
        if "=" in arg:
            name, value = arg.split("=")
            globals()[name.upper()] = int(value) if name == "offset" else float(value)
    for kind in kinds:
        totals = {name: Counter() for name in VERSIONS}
        for seed in range(cases):
            c = case(seed * 7919 + zlib.crc32(kind.encode()) % 1000 + OFFSET, kind, mishear)
            truth = Counter(spelling(i) for i, _a, _o in c.words)
            outs = {name: run(routes, c) for name, routes in VERSIONS.items()}
            m = Counter(outs["parakeet_main"][0])
            for name, (said, redos) in outs.items():
                t, p = totals[name], Counter(said)
                lost = sum(max(0, min(m[w], truth[w]) - p[w]) for w in truth)
                dup = sum(max(0, p[w] - max(m[w], truth[w])) for w in truth)
                invented = sum(max(0, p[w] - m[w]) for w in p if w not in truth)
                order = [int(w[1:]) for w in said if w in truth and truth[w] == 1 and w.startswith("w")]
                t["missing"] += sum(max(0, truth[w] - p[w]) for w in truth)
                t["lost"] += lost
                t["dup"] += dup
                t["invented"] += invented
                t["inverted"] += sum(1 for x, y in zip(order, order[1:]) if y < x)
                t["redos"] += redos
                if lost or dup or invented:
                    t["worse_cases"] += 1
                    if name == "parakeet_service" and "-v" in sys.argv:
                        print(f"  {kind} seed={c.seed} lost={lost} dup={[w for w in p if w in truth and p[w] > max(m[w], truth[w])]} invented={[w for w in p if w not in truth and p[w] > m[w]]}")
        print(kind, cases, "cases, mishear", mishear, "repeat/split/madeup/drift", REPEAT, SPLIT, MADEUP, DRIFT)
        for name, t in totals.items():
            print(f"  {name:18s} {dict(t)}")


if __name__ == "__main__":
    main()
