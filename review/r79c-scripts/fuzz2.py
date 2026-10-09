"""Identity-tracking fuzz: repeats, short function words, multi-token words,
misheard words, splits, jitter, skips in first decode and both redos.

WT=<wt> python fuzz2.py <seed> <n> <pieces:1|2|3|mix> [opts]
Prints one JSON line per case with each output word's identity."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import zlib

import lib
from lib import run

logging.disable(logging.CRITICAL)

SUF = "~^*+=<>|"


def enc(n):
    s = ""
    while True:
        s = SUF[n % 8] + s
        n //= 8
        if n == 0:
            return s


def dec(s):
    n = 0
    for ch in s:
        n = n * 8 + SUF.index(ch)
    return n


COMMON = ["the", "a", "and", "I", "no", "to", "of", "it"]

seed, n, mode = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
opts = dict(a.split("=") for a in sys.argv[4:])
P_JIT = float(opts.get("jit", 0.3))
P_MIS = float(opts.get("mis", 0.03))
P_SPLIT = float(opts.get("split", 0.0))
P_DEL = float(opts.get("del", 0.0))
P_RANGE_SKIP = float(opts.get("rangeskip", 0.6))
P_STRETCH_SKIP = float(opts.get("stretchskip", 0.2))
FRAG = opts.get("frag", "0") == "1"
CUTFOCUS = opts.get("cut", "0") == "1"


def identity(word):
    base = word.rstrip(SUF)
    suf = word[len(base):]
    if not suf:
        return None, base
    return dec(suf), base


def make_truth(rng, total, silences):
    truth, t, i = [], rng.uniform(0.0, 0.5), 0
    while t < total - 0.3:
        if any(a <= t < b for a, b in silences):
            t = next(b for a, b in silences if a <= t < b) + rng.uniform(0, 0.3)
            continue
        r = rng.random()
        if r < 0.12:  # a repeat run: "no no no", "the the", "I I I"
            base = rng.choice(COMMON)
            for _k in range(rng.choice([2, 2, 3])):
                truth.append((base, 1, round(t, 4)))
                t += rng.uniform(0.15, 0.35)
            continue
        if r < 0.4:
            base, ntok = rng.choice(COMMON), 1
        else:
            base, ntok = f"w{i}", rng.choice([1, 1, 2, 3])
        truth.append((base, ntok, round(t, 4)))
        i += 1
        t += rng.uniform(0.2, 0.6) + 0.06 * ntok
    words = []
    for idx, (base, ntok, t) in enumerate(truth):
        name = base + enc(idx)
        if ntok == 1:
            toks = [(" " + name, t)]
        else:
            parts = [base[: max(1, len(base) // ntok)]]
            rest = base[len(parts[0]):]
            while rest:
                parts.append(rest[: max(1, -(-len(rest) // (ntok - 1)))])
                rest = rest[len(parts[-1]):]
            parts[-1] += enc(idx)
            toks = [((" " if k == 0 else "") + p, round(t + 0.08 * k, 4)) for k, p in enumerate(parts)]
        words.append(lib.W(toks))
    return words


def crc(*xs):
    return zlib.crc32(repr(xs).encode())


for case in range(n):
    if os.environ.get("CASE") and int(os.environ["CASE"]) != case:
        continue
    rng = random.Random(seed * 1000003 + case)
    pieces = int(mode) if mode != "mix" else rng.choice([1, 2, 3])
    ctx = 5.0
    if pieces == 1:
        total = rng.uniform(20, 75)
        ranges, windows = [(0.0, total)], [(0.0, total)]
        speech = None
    elif pieces == 2:
        total = rng.uniform(90, 130)
        cut = rng.uniform(50, total - 30)
        ranges, windows = lib.two(total, cut, ctx)
        speech = [(0.0, total)]
    else:
        total = rng.uniform(150, 190)
        c1 = rng.uniform(50, 65)
        c2 = rng.uniform(c1 + 45, min(total - 25, c1 + 70))
        ranges, windows = lib.three(total, c1, c2, ctx)
        speech = [(0.0, total)]
    # a pause at each cut, as the chunker cuts in one
    cuts = [b for (_a, b) in ranges[:-1]]
    silences = [(c - rng.uniform(0.05, 0.4), c + rng.uniform(0.05, 0.4)) for c in cuts]
    truth = make_truth(rng, total, silences)

    def skips_for(a, b, phase, rng_):
        out = []
        k = rng_.choice([1, 1, 2])
        for _ in range(k):
            if CUTFOCUS and cuts and rng_.random() < 0.7:
                c = rng_.choice(cuts)
                if rng_.random() < 0.5:
                    lo, hi = c - rng_.uniform(3.5, 12), c + rng_.uniform(-1.5, 6)
                else:
                    lo, hi = c - rng_.uniform(-1.5, 6), c + rng_.uniform(3.5, 12)
            else:
                kind = rng_.choice(["start", "mid", "end"])
                length = rng_.uniform(3.5, 20)
                if kind == "start":
                    lo, hi = a, a + length
                elif kind == "end":
                    lo, hi = b - length, b
                else:
                    lo = rng_.uniform(a, max(a, b - length))
                    hi = lo + length
            out.append((max(a, lo), min(b, hi)))
        return out

    def beh(a, b, phase, idx):
        r = random.Random(crc(seed, case, phase, round(a, 3), round(b, 3)))
        if phase == "first":
            skip = skips_for(a, b, phase, r)
        elif phase == "_redo_ranges":
            skip = skips_for(a, b, phase, r) if r.random() < P_RANGE_SKIP else []
        else:
            skip = skips_for(a, b, phase, r) if r.random() < P_STRETCH_SKIP else []

        def jit(i, r_=crc(seed, case, phase, round(a, 3))):
            v = random.Random(crc(r_, i)).random()
            return 0 if v >= P_JIT else (-1 if v < P_JIT / 2 else 1)

        alt = {}
        for i, word in enumerate(truth):
            v = random.Random(crc(seed, case, phase, round(a, 3), i, "m")).random()
            base = word.name.rstrip(SUF)
            suf = word.name[len(base):]
            if v < P_MIS:
                alt[i] = [((" " if k == 0 else "") + ("x" + x.strip() if k == 0 else x), tt - word.t) for k, (x, tt) in enumerate(word.toks)]
            elif v < P_MIS + P_SPLIT:
                # heard as two words: the base, then a fragment "y" + base starting 0.16-0.4 s later
                d = random.Random(crc(seed, case, i, "d")).choice([0.16, 0.24, 0.32, 0.4])
                alt[i] = [(" " + base + suf, 0.0), (" y" + base + suf, d)]
            elif v < P_MIS + P_SPLIT + P_DEL and base in COMMON:
                alt[i] = []  # not heard
        return dict(skip=skip, jit=jit, alt=alt, frag=FRAG)

    try:
        res = run(total, ranges, windows, speech, lib.Model(truth, beh), vad=[(0.0, total)])
        ws = [(x["word"], round(x["start"], 3), round(x["end"], 3)) for x in res.words]
        seg = [s["segment"] for s in res.segments]
        chk = lib.check(res.words, res.text)
        chk["seg_mismatch"] = " ".join(seg).split() != [x["word"] for x in res.words]
        out = dict(case=case, pieces=pieces, ranges=ranges, truth=[x.name.rstrip(SUF) for x in truth], words=ws, calls=res.calls, **chk)
    except Exception as exc:  # noqa: BLE001
        import traceback
        out = dict(case=case, pieces=pieces, crash=repr(exc), tb=traceback.format_exc()[-800:])
    print(json.dumps(out))
