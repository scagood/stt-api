"""Quiet word VAD misses in a split first/last range's margin: decoded on main vs head.
Real volume VAD (file's own gate), parakeet-v2 bounds (25/30, ctx 5) unless CFG env.
usage: margin.py GEN(a|b) SIDE(tail|lead) MODE(fixed|random) N [LEVEL_DB]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from common import MAIN, HEAD, SR, plan
from gens import gen_a, gen_b, quiet_word
if os.environ.get('HEADMOD'):
    from common import load_local
    HEAD = load_local(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ['HEADMOD']), 'alt')

gen, side, mode, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
lvl_fixed = float(sys.argv[5]) if len(sys.argv) > 5 else None
cfg = os.environ.get("CFG", "v2 ctx5")
G = gen_a if gen == "a" else gen_b
T = Counter(); given = []; worst = []
for seed in range(N):
    r = np.random.default_rng(10_000 + seed * 7 + (0 if gen == "a" else 1) + (0 if side == "tail" else 2))
    if mode == "fixed":
        total_s = 40.0
        sp = (3.0, 37.0)
        w = (38.3, 38.7) if side == "tail" else (1.4, 1.8)
        lvl = -42.0 if lvl_fixed is None else lvl_fixed
    else:
        if side == "tail":
            lead = r.uniform(0.3, 3.0); L = r.uniform(36.0, 39.6); sp = (lead, lead + L)
            off = r.uniform(0.25, 2.9); wl = r.uniform(0.2, 0.45)
            w = (sp[1] + off, sp[1] + off + wl); total_s = w[1] + r.uniform(0.0, 3.0)
        else:
            lead = r.uniform(1.0, 3.5); L = r.uniform(36.0, 39.6)
            wl = r.uniform(0.2, 0.45); off = r.uniform(0.25, min(2.9, lead - wl))
            w = (lead - off - wl, lead - off); sp = (lead, lead + L); total_s = sp[1] + r.uniform(0.2, 3.0)
        lvl = r.uniform(-48, -38) if lvl_fixed is None else lvl_fixed
    total = int(total_s * SR)
    wav, words = G([(int(sp[0] * SR), int(sp[1] * SR))], total, r)
    a, b = int(w[0] * SR), int(w[1] * SR)
    quiet_word(wav, a, b, lvl, r, gen)
    res = {}
    for name, mod in (("main", MAIN), ("head", HEAD)):
        p = plan(mod, wav, cfg, gate=None)
        heard = any(s < b and e > a for s, e in p.speech)
        in_range = any(r0 <= a and b <= r1 for r0, r1 in p.ranges)
        in_win = any(w0 <= a and b <= w1 for w0, w1 in p.windows)
        part = any(r0 < b and a < r1 for r0, r1 in p.ranges)
        res[name] = (heard, in_range, in_win, part, p)
    T["n"] += 1
    T["vad_heard"] += res["main"][0]
    for k in ("main", "head"):
        T[k + "_range"] += res[k][1]; T[k + "_win"] += res[k][2]; T[k + "_partial"] += res[k][3] and not res[k][1]
    T["head_fewer"] += res["main"][1] and not res["head"][1]
    T["head_more"] += res["head"][1] and not res["main"][1]
    pm, ph = res["main"][4], res["head"][4]
    T["split"] += len(pm.ranges) > 1
    # margin given up (main's ranges minus head's), outside speech
    cm = np.zeros(total, bool); ch = np.zeros(total, bool)
    for r0, r1 in pm.ranges: cm[r0:r1] = True
    for r0, r1 in ph.ranges: ch[r0:r1] = True
    d = cm & ~ch
    if d.any():
        T["gave_up"] += 1
        idx = np.flatnonzero(d)
        given.append(d.sum() / SR)
        # each stretch given up, measured against the floor of the margin it lies in (lead or tail)
        over = -99.0
        for g0, g1 in MAIN.runs(d).tolist():
            if g1 <= pm.speech[0][0]:
                marg = wav[pm.ranges[0][0]: pm.speech[0][0]]
            elif g0 >= pm.speech[-1][1]:
                marg = wav[pm.speech[-1][1]: pm.ranges[-1][1]]
            else:
                T["given_inside_speech_span"] += 1; continue
            mr = MAIN.frame_rms(marg); floor = float(np.percentile(mr, 10))
            rms = MAIN.frame_rms(wav[g0:g1]) if g1 - g0 >= MAIN.FRAME else np.sqrt(np.mean(wav[g0:g1] ** 2))[None]
            over = max(over, 20 * np.log10(rms.max() / floor))
            T["given_lead" if g1 <= pm.speech[0][0] else "given_tail"] += 1
        worst.append(over)
        T["given_has_word"] += bool(d[a:b].any())
        T["given_touches_speechwords"] += any(d[s:e].any() for s, e in words)
    if res["main"][1] and not res["head"][1] and T["head_fewer"] <= 2:
        print(f"  seed {seed}: speech {sp[0]:.2f}-{sp[1]:.2f} word {w[0]:.2f}-{w[1]:.2f} @{lvl:.1f} dBFS VAD {[(round(s/SR,2), round(e/SR,2)) for s,e in pm.speech]}")
        print(f"    main {[(round(x/SR,2), round(y/SR,2)) for x,y in pm.ranges]}  head {[(round(x/SR,2), round(y/SR,2)) for x,y in ph.ranges]}")
g = np.array(given) if given else np.zeros(1)
wo = np.array(worst) if worst else np.zeros(1)
print(f"gen {gen} {side} {mode} {cfg} lvl={lvl_fixed}: n={T['n']} split={T['split']} VAD heard word {T['vad_heard']} | word in a range main {T['main_range']} head {T['head_range']} "
      f"(in window {T['main_win']}/{T['head_win']}, partial {T['main_partial']}/{T['head_partial']}) | head fewer {T['head_fewer']} more {T['head_more']} | "
      f"margin given up in {T['gave_up']} files, mean {g.mean():.2f}s max {g.max():.2f}s; loudest given-up frame over margin floor max {wo.max():.1f} dB "
      f"(median {np.median(wo):.1f}); given-up overlaps quiet word {T['given_has_word']}, speech words {T['given_touches_speechwords']}; stretches lead {T['given_lead']} tail {T['given_tail']} in-speech {T['given_inside_speech_span']}", flush=True)
