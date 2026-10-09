"""In-word share of forced cuts vs white-noise SNR, real speech (one segment, gate under the noise; and pauses
cut to 0.25 s with the file's own gate). usage: snr_sweep.py CH [CH...]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from synth import MAIN, PR, CONFIGS, SR, forced_cuts

DL = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/k80a-dl"
orig = {m: m._speech_segments for m in (MAIN, PR)}


def inw(x, words, widen=0.0):
    """inside a word's wav2vec2 span, each span widened by `widen` s a side (CTC spans are peaky)"""
    t = x / SR
    i = np.searchsorted(words[:, 0], t + widen, side="right") - 1
    return bool(i >= 0 and words[i, 0] - widen < t < words[i, 1] + widen)


def quiet(x, ref_wav):
    """the 80 ms around x (clean audio) 20 dB+ under the loudest tenth of the 4 s around it: a pause-like spot"""
    a, b = max(0, x - 2 * SR), min(ref_wav.size, x + 2 * SR)
    r = MAIN.frame_rms(ref_wav[a:b])
    here = float(np.sqrt(np.mean(ref_wav[max(0, x - 640): x + 640].astype(np.float64) ** 2)))
    return here < np.percentile(r, 90) * 0.1


TOT = {}
for ch in sys.argv[1:]:
    clean = np.fromfile(f"{DL}/holmes_{ch}.f32", dtype=np.float32)
    words = np.load(f"{DL}/words_{ch}.npy")
    rms = MAIN.frame_rms(clean)
    lvl = float(np.sqrt(np.mean(rms[rms > np.percentile(rms, 50)] ** 2)))
    keep = np.ones(clean.size, dtype=bool)
    for (s0, e0), (s1, e1) in zip(words[:-1], words[1:]):
        if s1 - e0 > 0.25:
            keep[int((e0 + 0.125) * SR): int((s1 - 0.125) * SR)] = False
    kb = np.concatenate(([0], np.cumsum(keep)))
    sqw = np.stack([kb[(words[:, 0] * SR).astype(int)], kb[(words[:, 1] * SR).astype(int)]], axis=1) / SR
    sq = clean[keep]
    rng = np.random.default_rng(100 + int("".join(c for c in ch if c.isdigit())))
    for snr in (12, 14, 16, 20, 30):
        for kind in ("one segment", "pauses 0.25 s"):
            base, w = (clean, words) if kind == "one segment" else (sq, sqw)
            ref = base
            amp = lvl * 10 ** (-snr / 20)
            wav = base + rng.standard_normal(base.size).astype(np.float32) * amp
            for cname in ("v2 ctx5", "v3 ctx5", "whisper"):
                cfg = CONFIGS[cname]
                T = TOT.setdefault((kind, snr, cname), Counter())
                out = []
                for mod in (MAIN, PR):
                    if kind == "one segment":
                        mod.VAD_GATE_DB = 20 * np.log10(amp) - 6
                    else:
                        mod.VAD_GATE_DB = None
                    mod._speech_segments = orig[mod]
                    p = mod.plan_chunks(wav, target_sec=cfg["target"], max_sec=cfg["mx"], min_sec=20.0, context_sec=cfg["ctx"])
                    out.append(forced_cuts(p.ranges, p.speech))
                fm, fp = out
                T["forced_main"] += len(fm); T["forced_pr"] += len(fp)
                T["main"] += sum(inw(x, w) for x in fm); T["pr"] += sum(inw(x, w) for x in fp)
                T["main_w50"] += sum(inw(x, w, 0.05) for x in fm); T["pr_w50"] += sum(inw(x, w, 0.05) for x in fp)
                T["main_q"] += sum(quiet(x, ref) for x in fm); T["pr_q"] += sum(quiet(x, ref) for x in fp)
                if len(fm) == len(fp):
                    for x, y in zip(fm, fp):
                        a, b = inw(x, w), inw(y, w)
                        T["g->W"] += (not a) and b; T["W->g"] += a and not b; T["moved"] += x != y
for (kind, snr, cname), T in sorted(TOT.items()):
    print(f"{kind:14s} SNR {snr:2d} {cname:8s} forced {T['forced_main']:4d}->{T['forced_pr']:4d} in word main {T['main']/max(1,T['forced_main']):4.0%} -> #80 {T['pr']/max(1,T['forced_pr']):4.0%}  "
          f"| spans +-50ms {T['main_w50']/max(1,T['forced_main']):4.0%} -> {T['pr_w50']/max(1,T['forced_pr']):4.0%} | at a pause-like spot (20 dB under) {T['main_q']/max(1,T['forced_main']):4.0%} -> {T['pr_q']/max(1,T['forced_pr']):4.0%} "
          f"(moved {T['moved']}, W->g {T['W->g']}, g->W {T['g->W']})", flush=True)
