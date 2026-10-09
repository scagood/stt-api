"""Adversarial fake-Parakeet fuzz for #79's identity splice, with repeated
function words, far-timed and misheard words, splits, drops and made-up words.

Drives the real routes._redo_stalled and routes._stitch. Every word carries an
identity tag made of non-word characters (which _seam_key strips), so repeats
like "the the" share a key but stay traceable.

python cutfz.py <worktree> <mode one|two|three|batch> <seed> <n> <out.jsonl> [k=v ...]
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import sys
import time
import traceback
import types
from types import SimpleNamespace

WT = sys.argv[1]
sys.path.insert(0, WT)
ort_stub = types.ModuleType("onnxruntime")
ort_stub.get_available_providers = lambda: ["CPUExecutionProvider"]
sys.modules.setdefault("onnx_asr", types.ModuleType("onnx_asr"))
sys.modules.setdefault("onnxruntime", ort_stub)

import numpy as np  # noqa: E402

from parakeet_service import routes  # noqa: E402
from parakeet_service.config import TARGET_SR as SR  # noqa: E402

logging.disable(logging.CRITICAL)
FRAME = 0.08
MODE, SEED, N, OUT = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
OPTS = dict(
    jit=0.25, far=0.03, endearly=0.15, drop=0.02, mis=0.05, split=0.03, madeup=0.25, madeup_start=0.05,
    shift=0.4, persist=0.5, fresh=0.1, func=0.45, rep=0.08, dense=0.0,
)
for kv in sys.argv[6:]:
    k, v = kv.split("=")
    OPTS[k] = float(v)

DIG = "#$%&*+=@"
FUNC = ["the", "a", "and", "I", "no", "to", "of", "it", "that", "you", "so", "uh"]


def tag(i):
    s = ""
    for _ in range(4):
        s = DIG[i % 8] + s
        i //= 8
    return s


def untag(word):
    t = word.rstrip("~")
    tail = t[-4:]
    if len(t) < 5 or any(c not in DIG for c in tail):
        return None
    v = 0
    for c in tail:
        v = v * 8 + DIG.index(c)
    return v


def gen_truth(rng, total):
    """[(idx, text, [token...], [abs time...])]"""
    words, t, i = [], rng.uniform(0.0, 0.6), 0
    while t < total - 0.3:
        if rng.random() < OPTS["func"]:
            text = rng.choice(FUNC)
            reps = 1 + (rng.random() < OPTS["rep"]) * rng.choice([1, 1, 2])
        else:
            text, reps = f"c{i}", 1
        for _r in range(reps):
            if t >= total - 0.3:
                break
            ntok = 1 if text in FUNC else rng.choices([1, 2, 3], [0.5, 0.3, 0.2])[0]
            parts = [text[: max(1, len(text) // ntok)]]
            rest = text[len(parts[0]):]
            while rest and len(parts) < ntok:
                parts.append(rest[: max(1, len(rest) // (ntok - len(parts)))])
                rest = rest[len(parts[-1]):]
            if rest:
                parts[-1] += rest
            toks = [" " + parts[0]] + parts[1:]
            times = [round(t + k * rng.choice([0.08, 0.16]), 4) for k in range(len(toks))]
            words.append((i, text, toks, times))
            i += 1
            r = rng.random()
            if OPTS["dense"]:
                gap = rng.uniform(0.12, 0.3)
            else:
                gap = rng.uniform(0.12, 0.4) if r < 0.6 else rng.uniform(0.4, 1.0) if r < 0.95 else rng.uniform(1.0, 4.0)
            t = max(t + gap, times[-1] + rng.choice([0.08, 0.16]))
    return words


def vad_of(words, total, music):
    segs = []
    for _i, _x, _toks, times in words:
        a, b = max(0.0, times[0] - 0.1), min(total, times[-1] + 0.3)
        if segs and a - segs[-1][1] < 0.4:
            segs[-1][1] = max(segs[-1][1], b)
        else:
            segs.append([a, b])
    segs += [list(m) for m in music]
    segs.sort()
    out = []
    for a, b in segs:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


class File:
    def __init__(self, rng, kind, fid):
        self.fid, self.kind = fid, kind
        total = {"one": rng.uniform(15, 75), "two": rng.uniform(80, 130), "three": rng.uniform(140, 190)}[kind]
        words = gen_truth(rng, total)
        music = []
        if rng.random() < 0.15:
            a = rng.uniform(0, total - 8)
            b = min(total, a + rng.uniform(5, 15))
            words = [w for w in words if not (a <= w[3][0] < b)]
            music.append((a, b))
        self.total, self.words, self.music = total, words, music
        self.vad = vad_of(words, total, music)
        ncuts = {"one": 0, "two": 1, "three": 2}[kind]
        cuts = []
        for c in range(ncuts):
            want = (c + 1) * total / (ncuts + 1) + rng.uniform(-6, 6)
            if rng.random() < 0.5:
                starts = [w[3][0] for w in words]
                ends = [w[3][-1] + 0.1 for w in words]
                best = min(range(1, len(words)), key=lambda k: abs((ends[k - 1] + starts[k]) / 2 - want))
                cut = round((ends[best - 1] + starts[best]) / 2, 3)
            else:
                cut = round(want, 3)
            cuts.append(cut)
        bounds = [0.0] + cuts + [round(total, 3)]
        ranges = [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]
        windows = []
        for k, (a, b) in enumerate(ranges):
            wa = a if k == 0 else max(ranges[k - 1][0], a - rng.uniform(3, 8))
            wb = b if k == len(ranges) - 1 else min(ranges[k + 1][1], b + rng.uniform(3, 8))
            windows.append((round(wa, 3), round(wb, 3)))
        self.ranges, self.windows = ranges, windows
        n = int(round(total * SR))
        self.wav = np.arange(n, dtype=np.float64) + fid * 1e8
        r = [(int(round(a * SR)), int(round(b * SR))) for a, b in ranges]
        w = [(int(round(a * SR)), int(round(b * SR))) for a, b in windows]
        r[-1] = (r[-1][0], n)
        w[-1] = (w[-1][0], n)
        speech = [(int(round(a * SR)), int(round(b * SR))) for a, b in self.vad]
        if kind == "one":
            speech = None if hasattr(routes, "speech_segments") else []
        self.prep = routes._PreparedAudio(
            waveform=self.wav, ranges=r, windows=w, speech=speech, pieces=[self.wav[a:b] for a, b in w], duration=total
        )
        self.first_skips, self.spots = [], []
        for ws, we in windows:
            k = rng.choices([0, 1, 2], [0.25, 0.5, 0.25])[0]
            sk = []
            for _ in range(k):
                ln = rng.uniform(3.5, 20)
                where = rng.choice(["start", "mid", "end", "cut"])
                if where == "start":
                    a = ws
                elif where == "end":
                    a = max(ws, we - ln)
                elif where == "cut" and len(ranges) > 1:
                    c = rng.choice([x for x in bounds[1:-1]])
                    a = c - rng.uniform(0, ln)
                else:
                    a = rng.uniform(ws, max(ws, we - ln))
                a = max(ws, a)
                b = min(we, a + ln)
                sk.append((a, b))
                if rng.random() < OPTS["persist"]:
                    self.spots.append((a, b, rng.uniform(8, 40)))
            self.first_skips.append(sk)


class Model:
    def __init__(self, case, files):
        self.case, self.files, self.calls = case, files, []

    def decode(self, f, ws, we, first_skips=None):
        rng = random.Random(f"{SEED}-{self.case}-{f.fid}-{ws:.4f}-{we:.4f}")
        if first_skips is None:
            self.calls.append((f.fid, round(ws, 3), round(we, 3)))
            skips = []
            for a, b, L in f.spots:
                lo, hi = max(a, ws), min(b, we)
                if we - ws >= L and hi - lo >= 0.5 * (b - a):
                    skips.append((lo, hi))
            if we - ws > 10 and rng.random() < OPTS["fresh"]:
                ln = rng.uniform(3.5, min(15, we - ws - 1))
                a = rng.uniform(ws, we - ln)
                skips.append((a, a + ln))
        else:
            skips = first_skips
        shift = rng.choice([-2, -1, 0, 1, 2]) if rng.random() < OPTS["shift"] else 0
        out = []  # (time, token)
        if rng.random() < OPTS["madeup_start"] and first_skips is None:
            out.append((rng.uniform(0, 0.3), " um"))
        for idx, text, toks, times in f.words:
            t0 = times[0]
            if not (ws <= t0 < we) or any(a <= t0 < b for a, b in skips):
                continue
            wr = random.Random(f"{SEED}-{self.case}-{f.fid}-{idx}-{ws:.4f}-{we:.4f}")
            if wr.random() < OPTS["drop"]:
                continue
            j = shift + (wr.choice([-1, 1]) if wr.random() < OPTS["jit"] else 0)
            if wr.random() < OPTS["far"]:
                j += wr.choice([-1, 1]) * wr.choice([3, 4, 5, 6])
            if we - t0 < 1.0 and wr.random() < OPTS["endearly"]:
                j -= wr.choice([4, 5, 6, 7])
            tg = tag(idx)
            o_toks, o_times = list(toks), list(times)
            o_toks[-1] = o_toks[-1] + tg
            r = wr.random()
            if r < OPTS["mis"]:
                o_toks, o_times = [" m" + text[::-1] + tg], [times[0]]
            elif r < OPTS["mis"] + OPTS["split"]:
                second = times[1] if len(times) > 1 else times[0] + wr.choice([0.16, 0.24, 0.32, 0.4, 0.48, 0.56])
                o_toks, o_times = [" s" + text[:1] + tg, " z" + text[1:] + tg + "~"], [times[0], second]
            for tok, t in zip(o_toks, o_times):
                if t >= we:
                    break
                out.append((max(0.0, (math.floor((t - ws) / FRAME + 1e-9) + j) * FRAME), tok))
        if first_skips is None and rng.random() < OPTS["madeup"]:
            at = (we - ws) - rng.uniform(0.0, 0.4)
            out.append((at, " uh"))
        # Parakeet's frames only go forward: keep emission order, clamp times
        tokens, stamps = [], []
        for t, tok in out:
            t = max(0.0, math.floor(t / FRAME + 1e-9) * FRAME)
            if stamps:
                t = max(t, stamps[-1])
            tokens.append(tok)
            stamps.append(round(t, 4))
        text = "".join(t.replace("▁", " ") for t in tokens).strip()
        return SimpleNamespace(text=text, tokens=tokens, timestamps=stamps)


class Worker:
    def __init__(self, model):
        self.model = model

    async def submit_many(self, pieces, _key):
        out = []
        for piece in pieces:
            fid = int(piece[0] // 1e8)
            f = self.model.files[fid]
            ws = (float(piece[0]) - f.fid * 1e8) / SR
            we = ws + piece.size / SR
            out.append(self.model.decode(f, ws, we))
        return out


def request(worker):
    state = SimpleNamespace(worker=worker, ready=True, audio_pool=None, align_pool=None)
    return SimpleNamespace(app=SimpleNamespace(state=state))


def problems_of(prep, out):
    text, segments, words = out
    p = []
    if text != routes._clean_text(" ".join(w["word"] for w in words)):
        p.append("text!=words")
    if text != routes._clean_text(" ".join(s["segment"] for s in segments)):
        p.append("text!=segments")
    starts = [w["start"] for w in words]
    if any(b < a - 1e-9 for a, b in zip(starts, starts[1:])):
        p.append("unsorted words")
    if any(w["end"] < w["start"] - 1e-9 for w in words):
        p.append("word end<start")
    if any(w["start"] < -1e-9 or w["end"] > prep.duration + 1e-6 for w in words):
        p.append("word outside clip")
    return p


def main():
    fout = open(OUT, "w")
    for case in range(N):
        rng = random.Random(f"{SEED}-{case}-layout")
        kinds = [rng.choice(["one", "two", "three"]) for _ in range(3)] if MODE == "batch" else [MODE]
        files = [File(rng, k, i) for i, k in enumerate(kinds)]
        model = Model(case, files)
        vads = {id(f.wav): [(int(round(a * SR)), int(round(b * SR))) for a, b in f.vad] for f in files}
        if hasattr(routes, "speech_segments"):
            routes.speech_segments = lambda wav, vads=vads: vads[id(wav)]
        rec = {"case": case, "kinds": kinds}
        try:
            first = []
            for f in files:
                for (a, b), sk in zip(f.windows, f.first_skips):
                    first.append(model.decode(f, a, b, first_skips=sk))
            t0 = time.perf_counter()
            results = asyncio.run(routes._redo_stalled(request(Worker(model)), [f.prep for f in files], first, "parakeet-v3:fp32"))
            rec["ms"] = (time.perf_counter() - t0) * 1000
            rec["calls"] = model.calls
            cursor, outs = 0, []
            for f in files:
                n = len(f.prep.pieces)
                o = routes._stitch(f.prep, results[cursor : cursor + n])
                cursor += n
                text, segments, words = o
                outs.append({
                    "words": [(w["word"], round(w["start"], 3)) for w in words],
                    "problems": problems_of(f.prep, o),
                    "truth": [(i, x, times[0]) for i, x, _t, times in f.words],
                    "ranges": f.ranges, "windows": f.windows, "first_skips": f.first_skips, "spots": f.spots,
                })
            rec["files"] = outs
        except Exception as exc:  # noqa: BLE001
            rec["crash"] = repr(exc)
            rec["tb"] = traceback.format_exc()[-2000:]
        fout.write(json.dumps(rec) + "\n")
    fout.close()


main()
