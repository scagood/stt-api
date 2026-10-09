import sys
import numpy as np
from parakeet_service import chunker
chunker.VAD = "volume"; chunker.VAD_GATE_DB = None
chunker._get_vad = lambda: (_ for _ in ()).throw(AssertionError)
SR = chunker.TARGET_SR

def syl(seconds, level_db, rng):
    """speech-like: 250 ms syllables / 100 ms dips (like tests' _turn) but Gaussian"""
    n = int(seconds * SR); x = rng.standard_normal(n) * 10 ** (level_db / 20)
    t = (np.arange(n) / SR) % 0.35
    return x * (t < 0.25)

def make(pause, events, speech_like=False, seed=0):
    rng = np.random.default_rng(seed)
    total = 80 + pause
    wav = rng.standard_normal(int(total * SR)) * 10 ** (-55 / 20)
    for a, b in ((0, 40), (40 + pause, 80 + pause)):
        wav[int(a*SR):int(b*SR)] += rng.standard_normal(int((b-a)*SR)) * 10 ** (-20 / 20)
    for at, dur, lvl, kind in events:
        n = int(dur * SR)
        if kind == "syl":
            seg = syl(dur, lvl, rng)
        elif kind == "yes":  # two syllables w/ a 60 ms dip in the middle
            seg = rng.standard_normal(n) * 10 ** (lvl / 20)
            m = n // 2; seg[m - int(0.03*SR): m + int(0.03*SR)] *= 0.1
        else:
            seg = rng.standard_normal(n) * 10 ** (lvl / 20)
        wav[int(at*SR):int(at*SR)+n] += seg
    return wav.astype(np.float32)

def plan(wav, ctx=0.0):
    p = chunker.plan_chunks(wav, target_sec=60, max_sec=75, context_sec=ctx)
    return [(round(a/SR, 2), round(b/SR, 2)) for a, b in p.ranges]

cases = {
 "click 10ms -30 @44.9 (10s pause)": (10, [(44.9, 0.01, -30, "n")]),
 "knock 30ms -35 @44.9 (10s pause)": (10, [(44.9, 0.03, -35, "n")]),
 "breath 300ms -42 @42.35 (5s pause)": (5, [(42.35, 0.3, -42, "n")]),
 "breath 300ms -42 @44.9 (10s pause)": (10, [(44.9, 0.3, -42, "n")]),
 "2 breaths 300ms -42 @42.5,@47 (10s)": (10, [(42.5, 0.3, -42, "n"), (47.0, 0.3, -42, "n")]),
 "3 breaths 250ms -42 @43,@48,@53 (15s)": (15, [(43, 0.25, -42, "n"), (48, 0.25, -42, "n"), (53, 0.25, -42, "n")]),
 "Yes 350ms -40 @44.8 (10s)": (10, [(44.8, 0.35, -40, "yes")]),
 "Yes 450ms -40 @44.8 (10s)": (10, [(44.8, 0.45, -40, "yes")]),
 "Yes 550ms -40 @44.8 (10s)": (10, [(44.8, 0.55, -40, "yes")]),
 "reply 700ms -40 @44.8 (10s)": (10, [(44.8, 0.70, -40, "yes")]),
 "Yes 350ms -40 + breath 300ms -42 (10s)": (10, [(44.8, 0.35, -40, "yes"), (47.5, 0.3, -42, "n")]),
 "quiet turn syl 2s -40 @44 (10s)": (10, [(44, 2.0, -40, "syl")]),
}
only = sys.argv[1:] 
for name, (pause, ev) in cases.items():
    print(f"{name:42s} {plan(make(pause, ev))}")
