"""Random layouts (c74r fuzz.py's generator) with synthesised speech inside the
segments: PR vs main on cuts inside words, pieces, bounds, and audio main
decoded that the PR no longer decodes."""
import random, zlib
from collections import Counter
from h import *


def layout(rng, scale=1.0):
    t = rng.uniform(0, 4)
    segs = []
    for _ in range(rng.randint(1, 8)):
        L = rng.choice([rng.uniform(0.3, 6), rng.uniform(5, 25), rng.uniform(15, 50)]) * scale
        segs.append((t, t + L))
        t += L + rng.choice([rng.uniform(0.05, 1.0), rng.uniform(0.5, 2.9), rng.uniform(3, 8)])
    total = segs[-1][1] + rng.choice([0, rng.uniform(0, 1.5), rng.uniform(1, 6)])
    return segs, total


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
    synth_kw = eval(os.environ.get("SYNTH", "{}"))
    only = os.environ.get("CFG")
    for cname, cfg in CONFIGS.items():
        if only and cname != only:
            continue
        rng = random.Random(f"{seed}|{cname}|{scale}")
        c = Counter()
        ex = {}
        for i in range(N):
            segs_s, total_s = layout(rng, scale)
            if total_s <= cfg["mx"]:
                continue
            segs = [(int(a * SR), int(b * SR)) for a, b in segs_s]
            total = int(total_s * SR)
            nrng = np.random.default_rng(zlib.crc32(f"{seed}|{cname}|{scale}|{i}".encode()))
            wav, words = synth(nrng, segs, total, **synth_kw)
            plans = {k: plan(M[k], wav, segs, cfg) for k in NAMES}
            rm, wm = plans["main"]
            c["layouts"] += 1
            nm, im = cuts_in_words(rm, segs, words)
            c["forced_main"] += nm; c["inword_main"] += im
            c["allcuts_inword_main"] += cuts_in_words_all(rm, words)
            for k in NAMES[1:]:
                rp, wp = plans[k]
                np_, ip = cuts_in_words(rp, segs, words)
                c[f"forced_{k}"] += np_; c[f"inword_{k}"] += ip
                c[f"allcuts_inword_{k}"] += cuts_in_words_all(rp, words)
                c[f"{k}_layouts_more_inword"] += ip > im
                c[f"{k}_diff_pieces"] += len(rp) != len(rm)
                bad = check_bounds(rp, wp, total, cfg)
                if bad:
                    c[f"{k}_bad"] += 1; ex.setdefault(f"{k} bad", (segs_s, total_s, bad))
                if min(b - a for a, b in rp) < cfg["mx"] * SR / 4 and min(b - a for a, b in rp) < min(b - a for a, b in rm):
                    c[f"{k}_part<max/4"] += 1
                gone = (covered(wm, total) & ~covered(wp, total)).sum() / SR
                if gone > 0:
                    c[f"{k}_layouts_lose_audio"] += 1
                    c[f"{k}_lose>0.5s"] += gone > 0.5
                    c[f"{k}_lose>1s"] += gone > 1.0
                    ex.setdefault(f"{k} lose", (segs_s, total_s, sec(rm), sec(rp), gone))
                if rm[0][0] < rp[0][0]: c[f"{k}_lead_shrank"] += 1
                if rm[-1][1] > rp[-1][1]: c[f"{k}_tail_shrank"] += 1
            c["oversized_layouts"] += nm > 0
        print(f"== {cname} scale {scale}: {dict(c)}")
        if c["forced_main"]:
            print("   in-word share (cuts inside VAD segments): " + ", ".join(f"{k} {c['inword_'+k]}/{c['forced_'+k]}={c['inword_'+k]/max(1,c['forced_'+k]):.3f}" for k in NAMES))
        for k, v in ex.items():
            print("   ex", k, v)
