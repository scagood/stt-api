import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/m75b-scripts")
from lib import *
import bisect

def syll(rng, level):
    L = int(rng.uniform(0.12, 0.25) * SR)
    return rng.standard_normal(L) * np.hanning(L) * db(level) * 1.6

def make(seed, b_level=-44, floor=-60, b_style="halting", turns=10):
    rng = np.random.default_rng(seed)
    parts, words, t = [], [], 0
    def add(x):
        nonlocal t
        parts.append(x); t += x.size
    def silence(sec):
        add(np.zeros(int(sec * SR)))
    silence(1.0)
    for turn in range(turns):
        # A: loud fluent 6-10 s
        end_t = t + int(rng.uniform(6, 10) * SR)
        while t < end_t:
            for w in range(int(rng.integers(2, 7))):
                ws = t
                for s in range(int(rng.integers(1, 4))):
                    add(syll(rng, -20)); silence(rng.uniform(0, 0.04))
                words.append([ws, t, "A"]); silence(rng.uniform(0.05, 0.15))
            silence(rng.uniform(0.3, 0.7))
        words[-1][2] = "A."
        silence(rng.uniform(1.0, 4.0))
        # B: quiet
        if b_style == "halting":
            n = int(rng.integers(2, 6))
            for w in range(n):
                ws = t
                for s in range(int(rng.integers(1, 3))):
                    add(syll(rng, b_level)); silence(rng.uniform(0, 0.04))
                words.append([ws, t, "b"]); silence(rng.uniform(0.5, 1.5))
            words[-1][2] = "b."
        else:
            end_t = t + int(rng.uniform(3, 8) * SR)
            while t < end_t:
                for w in range(int(rng.integers(2, 7))):
                    ws = t
                    for s in range(int(rng.integers(1, 4))):
                        add(syll(rng, b_level)); silence(rng.uniform(0, 0.04))
                    words.append([ws, t, "b"]); silence(rng.uniform(0.05, 0.15))
                silence(rng.uniform(0.3, 0.7))
            words[-1][2] = "b."
        silence(rng.uniform(3.5, 6.0))
    wav = np.concatenate(parts)
    wav = wav + rng.standard_normal(wav.size) * db(floor)
    # breaths 0.3 s at floor+12 dB in long silences
    truth = [{"word": w[2], "start": w[0] / SR, "end": w[1] / SR} for w in words]
    # Parakeet-ish slips
    para = []
    for i, w in enumerate(truth):
        s, e = w["start"] + rng.normal(0, 0.03), w["end"] + rng.normal(0, 0.03)
        if i and w["start"] - truth[i - 1]["end"] >= 0.3: s -= rng.uniform(0.1, 0.4)
        if i + 1 < len(truth) and truth[i + 1]["start"] - w["end"] >= 0.3: e += rng.uniform(0.1, 0.4)
        para.append({"word": w["word"], "start": max(0.0, s), "end": max(s + 0.04, e)})
    return wav.astype(np.float32), truth, para

def run(name, wav, para):
    ch, rt = MODS[name]
    p = ch.plan_chunks(wav, **BOUNDS, context_sec=5.0)
    gaps = rt.pauses(wav)
    ends = [g[1] for g in gaps]
    out = []
    for a, b in p.ranges:
        lo, hi = a / SR, b / SR
        cw = [w for w in para if lo <= w["start"] < hi]
        out += rt.retime(cw, rt.within(gaps, ends, lo, hi), lo, hi)
    return out, p.ranges

for style in ("halting", "fluent"):
    for b_level, floor in ((-40, -65), (-44, -60), (-46, -58)):
        agg = {n: [0, 0, 0, 0] for n in ("main", "prev", "head")}
        for seed in range(6):
            wav, truth, para = make(seed, b_level, floor, style)
            for n in agg:
                out, ranges = run(n, wav, para)
                got = {(round(w["start"], 6)): None for w in out}
                # match by index order within decoded words: decoded = words whose para start falls in a range
                dec = [i for i, w in enumerate(para) if any(a / SR <= w["start"] < b / SR for a, b in ranges)]
                lost_b = sum(1 for i, w in enumerate(truth) if w["word"].startswith("b") and i not in dec)
                errs = [abs(o["start"] - truth[i]["start"]) for i, o in zip(dec, out) if truth[i]["word"].startswith("b")]
                agg[n][0] += lost_b; agg[n][1] += sum(e > 0.3 for e in errs); agg[n][2] += sum(errs); agg[n][3] += len(errs)
        print(f"{style:8s} B{b_level}/floor{floor}: " + "  ".join(f"{n}: B lost {v[0]}, B moved>0.3s {v[1]}, mean|err| {v[2]/max(1,v[3]):.3f}" for n, v in agg.items()))
