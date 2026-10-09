"""Differential fuzz, main vs PR: quiet speech + non-speech in pauses between loud turns.
usage: fuzz.py N SEED MINSIL"""
import sys, json
import numpy as np
import lib

N, SEED, MINSIL = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
lib.set_minsil(MINSIL)
SR = lib.SR


def gen(rng):
    floor = rng.uniform(-62, -48)
    P = rng.uniform(5, 30)
    items = []  # (kind, start, end, [(at, len, dB)])
    t = rng.uniform(0.0, 3.0)
    spk = floor + rng.uniform(8, 25)
    while t < P - 0.2:
        kind = rng.choice(["phrase", "word", "breath", "train", "click", "word", "breath", "train"])
        parts = []
        if kind == "phrase":
            dur = rng.uniform(0.5, 2.0); u = t
            while u < t + dur:
                s = rng.uniform(0.1, 0.3); parts.append((u, s, spk + rng.uniform(-4, 3))); u += s + rng.uniform(0.03, 0.15)
            end = parts[-1][0] + parts[-1][1]
        elif kind == "word":
            u = t
            for _ in range(rng.integers(1, 3)):
                s = rng.uniform(0.08, 0.25); parts.append((u, s, spk + rng.uniform(-5, 3))); u += s + rng.uniform(0.03, 0.1)
            end = parts[-1][0] + parts[-1][1]
        elif kind == "breath":
            s = rng.uniform(0.08, 0.45); parts.append((t, s, floor + rng.uniform(5, 17))); end = t + s
        elif kind == "train":
            el = rng.uniform(0.04, 0.15); g = rng.uniform(0.2, 0.45); lv = floor + rng.uniform(8, 20); u = t
            for _ in range(rng.integers(2, 12)):
                parts.append((u, el, lv + rng.uniform(-2, 2))); u += el + g
            end = parts[-1][0] + parts[-1][1]
        else:
            s = rng.uniform(0.005, 0.03); parts.append((t, s, floor + rng.uniform(15, 30))); end = t + s
        if end > P - 0.05:
            break
        items.append((kind, t, end, parts))
        t = end + rng.uniform(0.1, 4.0)
    return floor, P, items


def evaluate(mod, wav, off, P, items):
    ch, rt = mod
    rg = lib.plan(mod, wav).ranges
    speech = [(off + a, off + b) for k, a, b, _ in items if k in ("phrase", "word")]
    dropped = cut = 0
    lost_s = 0.0
    status = []
    for a, b in speech:
        A, B = a * SR, b * SR
        covered = sum(max(0, min(y, B) - max(x, A)) for x, y in rg)
        whole = any(x <= A + 0.01 * SR and B - 0.01 * SR <= y for x, y in rg)
        status.append((2 if whole else (1 if covered > 0.01 * SR else 0), covered / SR, any(x < b - 0.04 and y > a + 0.04 for x, y in rt.pauses(wav)) if False else None))
        if not whole:
            # not wholly in one range
            covered = sum(max(0, min(y, B) - max(x, A)) for x, y in rg)
            if covered <= 0.01 * SR:
                dropped += 1
            else:
                cut += 1
            lost_s += (B - A - covered) / SR
    dec = sum(max(0, min(y / SR, off + P) - max(x / SR, off)) for x, y in rg)
    pz = rt.pauses(wav)
    inpause = sum(1 for a, b in speech if any(x < b - 0.04 and y > a + 0.04 for x, y in pz))
    loud = lib.loud_split(mod, wav)
    sp_frames = sum(int(loud[int(a / 0.02): int(b / 0.02)].sum()) for a, b in speech)
    return dict(dropped=dropped, cut=cut, lost=round(lost_s, 3), dec=round(dec, 2), inpause=inpause,
                spf=sp_frames, status=status, n=len(rg), ranges=[(int(x), int(y)) for x, y in rg])


rng = np.random.default_rng(SEED)
worse, better, more_dec, less_dec, same = [], [], [], [], 0
tot = {"main": dict(dropped=0, cut=0, inpause=0, lost=0.0, dec=0.0), "pr": dict(dropped=0, cut=0, inpause=0, lost=0.0, dec=0.0)}
nspeech = 0
for case in range(N):
    floor, P, items = gen(rng)
    sounds = [p for _, _, _, parts in items for p in parts]
    wav, off = lib.pause_between(P, sounds, floor=floor, seed=SEED * 100000 + case)
    m = evaluate(lib.MAIN, wav, off, P, items)
    p = evaluate(lib.PR, wav, off, P, items)
    nspeech += sum(1 for k, *_ in items if k in ("phrase", "word"))
    for k in tot["main"]:
        tot["main"][k] += m[k]; tot["pr"][k] += p[k]
    key = (p["dropped"] - m["dropped"], p["cut"] - m["cut"], round(p["lost"] - m["lost"], 3), p["inpause"] - m["inpause"])
    itemworse = [i for i, (s, t) in enumerate(zip(m["status"], p["status"])) if t[0] < s[0] or t[1] < s[1] - 0.02]
    if itemworse or p["inpause"] > m["inpause"]:
        key = key + (itemworse, [m["status"][i][:2] for i in itemworse], [p["status"][i][:2] for i in itemworse])
        worse.append((case, key, m["dec"], p["dec"]))
    elif any(x < 0 for x in key):
        better.append(case)
    if p["dec"] > m["dec"] + 0.05:
        more_dec.append((case, round(p["dec"] - m["dec"], 2)))
    elif p["dec"] < m["dec"] - 0.05:
        less_dec.append((case, round(m["dec"] - p["dec"], 2)))
print(f"minsil={MINSIL} seed={SEED} cases={N} speech items={nspeech}")
print(" totals main", {k: round(v, 2) for k, v in tot['main'].items()})
print(" totals PR  ", {k: round(v, 2) for k, v in tot['pr'].items()})
print(f" PR worse on speech: {len(worse)}  better: {len(better)}  PR decodes more pause: {len(more_dec)} (max {max([d for _, d in more_dec], default=0)} s)  less: {len(less_dec)}")
for w in worse[:15]:
    print("  worse", w)
