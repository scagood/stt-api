"""A quiet second speaker's fluent 10 s turn, syllables 0.15-0.3 s with 0.05-0.15 s dips,
each syllable at floor+lo..floor+hi dB, between two 40 s loud turns. Speech seconds decoded, per setting."""
import numpy as np
import lib

def case(seed, lo, hi, floor=-55):
    rng = np.random.default_rng(seed)
    sounds, spans = [], []
    t = 2.0
    while t < 12.0:
        s = rng.uniform(0.15, 0.3)
        sounds.append((t, s, floor + rng.uniform(lo, hi))); spans.append((t, t + s)); t += s + rng.uniform(0.05, 0.15)
        if rng.random() < 0.15:
            t += rng.uniform(0.2, 0.8)   # a gap between words
    wav, off = lib.pause_between(16.0, sounds, floor=floor, seed=seed + 7)
    return wav, off, spans

for lo, hi in ((6, 12), (8, 14), (10, 16)):
    for ms in (100, 400, 700, 1000, 2000):
        lib.set_minsil(ms)
        res = {}
        for name, mod in (("main", lib.MAIN), ("PR", lib.PR)):
            tot = got = 0.0
            for seed in range(40):
                wav, off, spans = case(seed, lo, hi)
                rg = lib.plan(mod, wav).ranges
                for a, b in spans:
                    A, B = (off + a) * lib.SR, (off + b) * lib.SR
                    tot += b - a
                    if any(x <= A and B <= y for x, y in rg):
                        got += b - a
            res[name] = got / tot
        print(f"syllables floor+{lo}..{hi} dB  min_silence {ms:4d}: speech decoded main {res['main']:.0%}  PR {res['PR']:.0%}")
