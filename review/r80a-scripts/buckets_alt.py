"""In-word forced cuts, main vs head, on an independent generator (VAD patched to the layout's phrases).
Generator C: phrases of words made of 1-3 syllables (cosine-ramped plateaus) with syllable dips,
level = phrase level + slow drift (random walk, ~2 dB) + per-word offset; lognormal gaps (median 120 ms, 30-390 ms,
25% of them 30-70 ms); pink room tone -62 dBFS in pauses; optional stop closures.
usage: buckets.py BUCKET N SEED [scales]"""
import os, sys, bisect, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from common import MAIN, HEAD, SR, plan, CONFIGS
from gens import pink
if os.environ.get('HEADMOD'):
    from common import load_local
    HEAD = load_local(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ['HEADMOD']), 'alt')

BUCKETS = {
    "gap6":  dict(gap=(5, 7),   word=4, dips=(6, 10), clos=0.0, clos_db=(25, 35)),
    "gap10": dict(gap=(9, 11),  word=4, dips=(6, 10), clos=0.0, clos_db=(25, 35)),
    "gap20": dict(gap=(18, 22), word=4, dips=(6, 10), clos=0.0, clos_db=(25, 35)),
    "gap30": dict(gap=(28, 32), word=4, dips=(6, 10), clos=0.0, clos_db=(25, 35)),
    "clos40_gap20": dict(gap=(18, 22), word=4, dips=(6, 10), clos=0.4, clos_db=(25, 35)),
    "clos50_gap8-15": dict(gap=(8, 15), word=4, dips=(6, 10), clos=0.5, clos_db=(25, 35)),
    "uneven4_gap10": dict(gap=(8, 12), word=4, dips=(6, 10), clos=0.0, clos_db=(25, 35), uniform=True),
    "uneven8_gap10": dict(gap=(8, 12), word=8, dips=(6, 10), clos=0.0, clos_db=(25, 35), uniform=True),
    "uneven8_gap6": dict(gap=(5, 7), word=8, dips=(6, 10), clos=0.0, clos_db=(25, 35), uniform=True),
    "uneven8_gap20": dict(gap=(18, 22), word=8, dips=(6, 10), clos=0.0, clos_db=(25, 35), uniform=True),
    "flat_gap6": dict(gap=(5, 7), word=0.5, dips=(0, 0), clos=0.0, clos_db=(25, 35)),
}


def layout(rng, scale):
    segs, t = [], rng.uniform(0.2, 3.5)
    n = int(rng.integers(1, 7))
    for i in range(n):
        L = (rng.uniform(20, 120) if rng.random() < 0.45 else rng.uniform(1, 8)) * scale
        segs.append((t, t + L))
        t += L + (rng.uniform(0.45, 1.5) if rng.random() < 0.6 else rng.uniform(1.5, 3.2) if rng.random() < 0.7 else rng.uniform(3.2, 8))
    total = segs[-1][1] + rng.uniform(0.1, 4.0)
    return segs, total


def synth(segs, total, rng, B):
    N = int(total * SR)
    wav = pink(N, rng) * 10 ** (-62 / 20)
    words, inner = [], []
    for s0, e0 in segs:
        s, e = int(s0 * SR), int(e0 * SR)
        phrase = rng.uniform(-30, -14)
        # slow drift: random walk sampled every 0.5 s, sd 0.7 dB/step, clipped to +-4 dB
        steps = int((e - s) / SR / 0.5) + 2
        drift = np.clip(np.cumsum(rng.normal(0, 0.7, steps)), -4, 4)
        t = s
        while t < e:
            wl = int(rng.uniform(0.12, 0.6) * SR); we = min(e, t + wl); n = we - t
            if n >= 64:
                base = phrase + drift[int((t - s) / SR / 0.5)]
                off = rng.uniform(-B["word"], B["word"]) if B.get("uniform") else rng.normal(0, B["word"] / 2)
                lvl = 10 ** ((base + off) / 20)
                env = np.ones(n, np.float32)
                nsyl = 1 if n < 0.18 * SR else int(rng.integers(1, 4))
                if B["dips"][1] > 0:
                    for k in range(1, nsyl):
                        c = n * k // nsyl + int(rng.uniform(-0.15, 0.15) * n / nsyl)
                        w = int(rng.uniform(0.02, 0.06) * SR)
                        env[max(0, c - w // 2): c + w // 2] = 10 ** (-rng.uniform(*B["dips"]) / 20)
                if B["clos"] and rng.random() < B["clos"] and n > int(0.18 * SR):
                    w = int(rng.uniform(0.05, 0.12) * SR)
                    c = int(rng.uniform(0.25, 0.75) * n)
                    env[max(int(0.02 * SR), c - w // 2): min(n - int(0.02 * SR), c + w // 2)] = 10 ** (-rng.uniform(*B["clos_db"]) / 20)
                r = min(int(rng.uniform(0.01, 0.03) * SR), n // 3)
                ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, r))).astype(np.float32)
                env[:r] *= np.maximum(ramp, 0.05); env[n - r:] *= np.maximum(ramp[::-1], 0.05)
                # light amplitude flutter inside the word (+-1.5 dB at 4-7 Hz)
                tt = np.arange(n) / SR
                env *= (10 ** (1.5 * np.sin(2 * np.pi * rng.uniform(4, 7) * tt + rng.uniform(0, 6.28)) / 20)).astype(np.float32)
                wav[t:we] = rng.standard_normal(n).astype(np.float32) * lvl * env
                words.append((t, we))
            gsec = rng.uniform(0.03, 0.07) if rng.random() < 0.25 else float(np.clip(rng.lognormal(np.log(0.12), 0.55), 0.03, 0.39))
            ge = min(e, we + int(gsec * SR))
            if ge > we:
                gl = 10 ** ((phrase + drift[min(len(drift) - 1, int((we - s) / SR / 0.5))] - rng.uniform(*B["gap"]) + rng.normal(0, 1)) / 20)
                wav[we:ge] = rng.standard_normal(ge - we).astype(np.float32) * gl + wav[we:ge]
            t = ge
    return wav, words


def inside(x, spans, starts):
    i = bisect.bisect_right(starts, x) - 1
    return i >= 0 and spans[i][0] < x < spans[i][1]


def evaluate(p, segs, total, cfg, words, wstarts, sstarts):
    t, mx, ctx = CONFIGS[cfg]
    own = int((mx - 2 * ctx) * SR); M = int(mx * SR)
    R, W = p.ranges, p.windows
    cuts = [b for (a, b), (c, d) in zip(R, R[1:]) if b == c]
    forced = [c for c in cuts if inside(c, segs, sstarts)]
    bad = sum(not (0 <= a < b <= total) for a, b in R) + sum(b > c for (a, b), (c, d) in zip(R, R[1:]))
    bad += sum(b - a > own for a, b in R) + sum(b - a > M for a, b in W) + sum(not (wa <= a and b <= wb) for (a, b), (wa, wb) in zip(R, W))
    lost = 0
    for s, e in segs:
        lost += (e - s) - sum(max(0, min(b, e) - max(a, s)) for a, b in R)
    # context edges (window edges that are not range edges)
    edges = [wa for (a, b), (wa, wb) in zip(R, W) if wa < a] + [wb for (a, b), (wa, wb) in zip(R, W) if wb > b]
    return dict(
        n=len(R), forced=len(forced), inword=sum(inside(c, words, wstarts) for c in forced),
        deep=sum(1 for c in forced if inside(c, words, wstarts) and min(c - words[bisect.bisect_right(wstarts, c) - 1][0], words[bisect.bisect_right(wstarts, c) - 1][1] - c) > 0.04 * SR),
        short=sum(b - a < 2 * SR for a, b in R), silent=sum(not any(s < b and e > a for s, e in segs) for a, b in R),
        lost=lost, bad=bad, minpart=min(b - a for a, b in R),
        edges=len(edges), edge_speech=sum(inside(x, segs, sstarts) for x in edges), edge_word=sum(inside(x, words, wstarts) for x in edges),
        fc=forced, R=R,
    )


def main():
    bname, N, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    scales = [float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1.0, 1.6]
    B = BUCKETS[bname]
    out = {}
    for scale in scales:
        rng = np.random.default_rng([seed, int(scale * 10), sum(map(ord, bname))])
        T = {c: Counter() for c in CONFIGS}
        for i in range(N):
            segs_s, total_s = layout(rng, scale)
            total = int(total_s * SR)
            segs = [(int(a * SR), int(b * SR)) for a, b in segs_s]
            wav, words = synth(segs_s, total_s, rng, B)
            wstarts = [w[0] for w in words]; sstarts = [s[0] for s in segs]
            for cfg in CONFIGS:
                if total_s <= CONFIGS[cfg][1]:
                    continue
                a = evaluate(plan(MAIN, wav, cfg, segs), segs, total, cfg, words, wstarts, sstarts)
                b = evaluate(plan(HEAD, wav, cfg, segs), segs, total, cfg, words, wstarts, sstarts)
                C = T[cfg]
                C["layouts"] += 1
                C["chunked_forced"] += a["forced"] > 0
                for k in ("n", "forced", "inword", "deep", "short", "silent", "lost", "bad", "edges", "edge_speech", "edge_word"):
                    C["m_" + k] += a[k]; C["h_" + k] += b[k]
                for k in ("n", "inword", "short", "silent", "lost", "bad", "forced"):
                    C["worse_" + k] += b[k] > a[k]; C["better_" + k] += b[k] < a[k]
                C["edge_word_worse"] += b["edge_word"] > a["edge_word"]; C["edge_word_better"] += b["edge_word"] < a["edge_word"]
                M = int(CONFIGS[cfg][1] * SR) - 2 * int(CONFIGS[cfg][2] * SR)
                C["quarter_viol"] += b["minpart"] < min(a["minpart"], M // 4)
                # coverage main has that head lacks
                cm = np.zeros(total, bool); ch = np.zeros(total, bool)
                for r0, r1 in a["R"]: cm[r0:r1] = True
                for r0, r1 in b["R"]: ch[r0:r1] = True
                d = cm & ~ch
                if d.any():
                    C["drop_layouts"] += 1; C["drop_samples"] += int(d.sum())
                    C["drop_in_word"] += sum(int(d[s:e].any()) for s, e in words)
                    C["drop_in_speech"] += sum(int(d[s:e].any()) for s, e in segs)
        for cfg, C in T.items():
            out[f"{bname} x{scale} {cfg}"] = dict(C)
            f = C.__getitem__
            print(f"{bname:15s} x{scale} {cfg}: L={f('layouts')} pieces {f('m_n')}->{f('h_n')} forced {f('m_forced')}->{f('h_forced')} "
                  f"inword {f('m_inword')} ({f('m_inword')/max(1,f('m_forced')):.1%}) -> {f('h_inword')} ({f('h_inword')/max(1,f('h_forced')):.1%}) "
                  f"deep {f('m_deep')}->{f('h_deep')} | worse {f('worse_inword')} better {f('better_inword')} | <2s {f('m_short')}->{f('h_short')} "
                  f"silent {f('m_silent')}->{f('h_silent')} lost {f('m_lost')}->{f('h_lost')} bad {f('m_bad')}->{f('h_bad')} q {f('quarter_viol')} "
                  f"worse n/short/silent/lost {f('worse_n')}/{f('worse_short')}/{f('worse_silent')}/{f('worse_lost')} | drop {f('drop_layouts')} "
                  f"({f('drop_samples')/SR:.1f}s, in word {f('drop_in_word')}, in speech {f('drop_in_speech')}) | ctx edges {f('m_edges')}->{f('h_edges')} "
                  f"in speech {f('m_edge_speech')}->{f('h_edge_speech')} in word {f('m_edge_word')} ({f('m_edge_word')/max(1,f('m_edges')):.1%}) -> {f('h_edge_word')} ({f('h_edge_word')/max(1,f('h_edges')):.1%}) "
                  f"edge-word worse/better {f('edge_word_worse')}/{f('edge_word_better')}", flush=True)
    json.dump(out, open(f"{os.path.dirname(os.path.abspath(__file__))}/out_{bname}_{seed}{os.environ.get('HEADMOD', '')}.json", "w"))


if __name__ == "__main__":
    main()
