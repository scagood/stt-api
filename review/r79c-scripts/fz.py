"""Fake-Parakeet fuzz driving the real routes._redo_stalled and routes._stitch.

python fz.py <worktree> <mode one|two|three|batch> <seed> <n> <out.jsonl> [opts as k=v]
opts: jit=0.1 mis=0.07 madeup=0.0 persist=0.35 fresh=0.1 paths=10

Every decode is a deterministic function of (seed, case, window), so main and
head see the same answers for the same windows.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
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

from parakeet_service import aligner, routes  # noqa: E402
from parakeet_service.config import TARGET_SR as SR  # noqa: E402

logging.disable(logging.CRITICAL)
FRAME = 0.08
MODE, SEED, N, OUT = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
OPTS = dict(jit=0.1, mis=0.07, madeup=0.0, persist=0.35, fresh=0.1, paths=10, mmid=0.0, mstart=0.0)
for kv in sys.argv[6:]:
    k, v = kv.split("=")
    OPTS[k] = float(v)

aligner.ALIGN_DEFAULT_LANGUAGE = "en"


class FakeChunk:
    def __init__(self, wav):
        self.size = wav.size

    def spans(self, words):
        n, dur = len(words), self.size / SR
        return [(dur * i / n, dur * (i + 0.5) / n) for i in range(n)]

    def scores(self, options, _s, _e):
        return [0.0] * len(options)

    def best(self, _o, _s, _e):
        return 0


aligner.for_chunk = lambda wav, language, name=None, quantization=None: FakeChunk(wav)


def gen_truth(rng, total, start=0.0):
    """[(idx, [token...], [abs time...])]"""
    words, t, i = [], start + rng.uniform(0.0, 0.6), 0
    while t < total - 0.3:
        ntok = rng.choices([1, 2, 3], [0.55, 0.3, 0.15])[0]
        toks = [f" w{i}"] + ["x", "y"][: ntok - 1]
        if rng.random() < 0.08:
            toks[-1] += ","  # punctuation on the word: same _seam_key
        times = [round(t + k * rng.choice([0.08, 0.16]), 4) for k in range(ntok)]
        words.append((i, toks, times))
        i += 1
        r = rng.random()
        gap = rng.uniform(0.2, 0.5) if r < 0.6 else rng.uniform(0.5, 1.0) if r < 0.95 else rng.uniform(1.0, 4.0)
        t = max(t + gap, times[-1] + rng.choice([0.08, 0.16, 0.24]))
    return words


def vad_of(words, total, music):
    segs = []
    for _i, _toks, times in words:
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
        if kind == "one":
            total = rng.uniform(15, 75)
        elif kind == "two":
            total = rng.uniform(90, 140)
        else:
            total = rng.uniform(150, 200)
        words = gen_truth(rng, total)
        music = []
        if rng.random() < 0.2:  # a music bed / noise VAD calls speech, with no words
            a = rng.uniform(0, total - 8)
            b = min(total, a + rng.uniform(5, 20))
            words = [w for w in words if not (a <= w[2][0] < b)]
            music.append((a, b))
        self.total, self.words, self.music = total, words, music
        self.vad = vad_of(words, total, music)
        ncuts = {"one": 0, "two": 1, "three": 2}[kind]
        cuts, gaps = [], []
        for c in range(ncuts):
            want = (c + 1) * total / (ncuts + 1) + rng.uniform(-8, 8)
            if rng.random() < 0.65:  # mid-pause, as the chunker cuts
                starts = [w[2][0] for w in words]
                ends = [w[2][-1] + 0.1 for w in words]
                best = min(range(1, len(words)), key=lambda k: abs((ends[k - 1] + starts[k]) / 2 - want))
                cut = round((ends[best - 1] + starts[best]) / 2, 3)
            else:  # a forced cut, maybe mid-word
                cut = round(want, 3)
            cuts.append(cut)
            gaps.append(False)
        ranges, windows = [], []
        bounds = [0.0] + cuts + [round(total, 3)]
        for k in range(len(bounds) - 1):
            ranges.append((bounds[k], bounds[k + 1]))
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
        if kind == "one":  # as plan_chunks leaves it: no VAD (None on the PR, [] on main)
            speech = None if hasattr(routes, "speech_segments") else []
        self.prep = routes._PreparedAudio(
            waveform=self.wav, ranges=r, windows=w, speech=speech, pieces=[self.wav[a:b] for a, b in w], duration=total
        )
        # skips for the first decodes, and which persist for redo decodes
        self.first_skips = []
        self.spots = []  # (a, b, L): a redo of at least L s covering half of it skips it again
        for ws, we in windows:
            k = rng.choices([0, 1, 2], [0.3, 0.45, 0.25])[0]
            sk = []
            for _ in range(k):
                ln = rng.uniform(3.5, 25)
                where = rng.choice(["start", "mid", "end"])
                if where == "start":
                    a = ws
                elif where == "end":
                    a = max(ws, we - ln)
                else:
                    a = rng.uniform(ws, max(ws, we - ln))
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
        tokens, stamps = [], []
        for idx, toks, times in f.words:
            t0 = times[0]
            if not (ws <= t0 < we) or any(a <= t0 < b for a, b in skips):
                continue
            wr = random.Random(f"{SEED}-{self.case}-{f.fid}-{idx}-{ws:.4f}-{we:.4f}")
            j = wr.choice([-1, 1]) if wr.random() < OPTS["jit"] else 0
            out_toks, out_times = list(toks), list(times)
            if wr.random() < OPTS["mis"]:
                kind = wr.choices(["other", "split", "surface"], [0.6, 0.25, 0.15])[0]
                if kind == "other":
                    out_toks = [f" m{idx}"]
                    out_times = [times[0]]
                elif kind == "split":
                    second = times[1] if len(times) > 1 else times[0] + wr.choice([0.16, 0.24])
                    out_toks, out_times = [f" m{idx}a", f" m{idx}b"], [times[0], second]
                else:
                    out_toks = [f" W{idx}."]
                    out_times = [times[0]]
            for tok, t in zip(out_toks, out_times):
                if t >= we:
                    break
                rel = max(0.0, (math.floor((t - ws) / FRAME + 1e-9) + j) * FRAME)
                if stamps:
                    rel = max(rel, stamps[-1])  # Parakeet's frames only go forward
                tokens.append(tok)
                stamps.append(round(rel, 4))
        if first_skips is None and rng.random() < OPTS["madeup"]:
            if rng.random() < OPTS["mmid"]:
                # made up inside the input's last 1.5 s, real words may follow it
                at = round(math.floor(max(0.0, (we - ws) - rng.uniform(0.25, 1.5)) / FRAME) * FRAME, 4)
                k = 0
                while k < len(stamps) and stamps[k] <= at:
                    k += 1
                tokens.insert(k, " uh")
                stamps.insert(k, at)
            else:
                at = max((stamps[-1] + FRAME) if stamps else 0.0, (we - ws) - rng.uniform(0.0, 0.3))
                tokens.append(" uh")
                stamps.append(round(math.floor(at / FRAME) * FRAME, 4))
        if first_skips is None and rng.random() < OPTS["mstart"]:
            tokens.insert(0, " um")
            stamps.insert(0, 0.0)
        text = "".join(t.replace("▁", " ") for t in tokens).strip()
        return SimpleNamespace(text=text, tokens=tokens, timestamps=stamps)


class Worker:
    def __init__(self, model):
        self.model = model

    def owner(self, piece):
        fid = int(piece[0] // 1e8)
        return self.model.files[fid]

    async def submit_many(self, pieces, _key):
        out = []
        for piece in pieces:
            f = self.owner(piece)
            ws = (float(piece[0]) - f.fid * 1e8) / SR
            we = ws + piece.size / SR
            out.append(self.model.decode(f, ws, we))
        return out


def request(worker):
    state = SimpleNamespace(worker=worker, ready=True, audio_pool=None, align_pool=None)
    return SimpleNamespace(app=SimpleNamespace(state=state))


def summarize(prep, out):
    text, segments, words = out
    return {
        "text": text,
        "segs": [(round(s["start"], 3), round(s["end"], 3), s["segment"]) for s in segments],
        "words": None if words is None else [(w["word"], round(float(w["start"]), 3), round(float(w["end"]), 3)) for w in words],
        "srt": routes._segments_to_srt(segments),
        "vtt": routes._segments_to_vtt(segments),
    }


def check_formats(prep, out):
    text, segments, words = out
    problems = []
    if words is not None and text != routes._clean_text(" ".join(w["word"] for w in words)):
        problems.append("text!=words")
    if text != routes._clean_text(" ".join(s["segment"] for s in segments)):
        problems.append("text!=segments")
    srt, vtt = routes._segments_to_srt(segments), routes._segments_to_vtt(segments)
    for s in segments:
        if s["segment"].strip() and (s["segment"].strip() not in srt or s["segment"].strip() not in vtt):
            problems.append("srt/vtt missing segment")
            break
    vj = routes._verbose_json("en", prep.duration, text, segments, words)
    if vj.get("text") != text:
        problems.append("verbose_json text")
    if words is not None:
        starts = [w["start"] for w in words]
        if any(b < a - 1e-9 for a, b in zip(starts, starts[1:])):
            problems.append("unsorted words")
        if any(w["end"] < w["start"] - 1e-9 for w in words):
            problems.append("word end<start")
    segst = [s["start"] for s in segments]
    if any(b < a - 1e-9 for a, b in zip(segst, segst[1:])):
        problems.append("unsorted segments")
    return problems


def main():
    fout = open(OUT, "w")
    for case in range(N):
        rng = random.Random(f"{SEED}-{case}-layout")
        if MODE == "batch":
            kinds = [rng.choice(["one", "one", "two"]) for _ in range(3)]
        else:
            kinds = [MODE]
        files = [File(rng, k, i) for i, k in enumerate(kinds)]
        model = Model(case, files)
        # VAD for one-piece clips (head runs it inside _redo_stalled)
        vads = {id(f.wav): [(int(round(a * SR)), int(round(b * SR))) for a, b in f.vad] for f in files}
        vad_calls = []

        def fake_vad(wav, vads=vads, vad_calls=vad_calls):
            vad_calls.append(wav.size)
            return vads[id(wav)]

        if hasattr(routes, "speech_segments"):
            routes.speech_segments = fake_vad
        rec = {"case": case, "kinds": kinds}
        try:
            first = []
            for f in files:
                for (a, b), sk in zip(f.windows, f.first_skips):
                    first.append(model.decode(f, a, b, first_skips=sk))
            worker = Worker(model)
            t0 = time.perf_counter()
            results = asyncio.run(routes._redo_stalled(request(worker), [f.prep for f in files], first, "parakeet-v3:fp32"))
            rec["redo_ms"] = (time.perf_counter() - t0) * 1000
            rec["calls"] = model.calls
            rec["vad_calls"] = len(vad_calls)
            cursor, outs = 0, []
            for f in files:
                n = len(f.prep.pieces)
                res = results[cursor : cursor + n]
                cursor += n
                o = routes._stitch(f.prep, res)
                fo = summarize(f.prep, o)
                fo["problems"] = check_formats(f.prep, o)
                fo["truth"] = [(idx, "".join(t).strip(), times[0]) for idx, t, times in f.words]
                fo["ranges"], fo["windows"], fo["first_skips"] = f.ranges, f.windows, f.first_skips
                fo["spots"] = f.spots
                if OPTS["paths"] and case % int(OPTS["paths"]) == 0:
                    paths = {}
                    for name, kw in [
                        ("speak", dict(speak=True, language="en")),
                        ("align", dict(align=True, language="en", aligner_choice=("wav2vec2-base-960h", "int8"))),
                        ("retime", dict(retime_words=True)),
                    ]:
                        try:
                            po = routes._stitch(f.prep, res, **kw)
                            paths[name] = {"problems": check_formats(f.prep, po), "n": len(po[2] or []),
                                           "words": [w["word"] for w in po[2]] if po[2] is not None else None}
                        except Exception as exc:  # noqa: BLE001
                            paths[name] = {"crash": repr(exc)}
                    fo["paths"] = paths
                outs.append(fo)
            rec["files"] = outs
        except Exception as exc:  # noqa: BLE001
            rec["crash"] = repr(exc)
            rec["tb"] = traceback.format_exc()[-1500:]
        fout.write(json.dumps(rec) + "\n")
    fout.close()


main()
