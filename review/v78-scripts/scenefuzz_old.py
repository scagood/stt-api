"""Random scenes: loud turns with internal pauses, and pauses holding quiet phrases, quiet words,
breaths and short trains of clicks/steps (<400 ms apart). Compare main vs PR per scene:
ground-truth speech samples decoded, cuts and window edges inside speech, non-speech decoded,
range count, and retime pauses overlapping speech. Usage: fuzz.py N seed [min_silence_ms]"""
import sys

import numpy as np

import lib_b as lib

SR = lib.SR
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
MS = int(sys.argv[3]) if len(sys.argv) > 3 else 400
for c in (lib.CO, lib.CP):
    c.VAD_MIN_SILENCE_MS = MS
DENSE = "dense" in sys.argv
STRADDLE = "straddle" in sys.argv


def scene(rng):
    floor = rng.uniform(-62, -50)
    parts, speech, nonspeech = [], [], []
    t = 0.0

    def emit(buf):
        nonlocal t
        parts.append(buf)
        t += buf.size / SR

    while t < 150:
        # loud turn with internal pauses
        L = rng.uniform(5, 40)
        buf = lib.noise(L, floor, rng)
        x = 0.0
        while x < L - 0.3:
            ph = min(rng.uniform(1.0, 6.0), L - x)
            seg = lib.turn(ph, -20, rng)
            i = int(x * SR)
            buf[i:i + seg.size] += seg[: buf.size - i]
            speech.append((t + x, t + x + ph))
            x += ph + rng.uniform(0.45, 1.5)
        emit(buf)
        # pause
        P = rng.uniform(3, 16)
        buf = lib.noise(P, floor, rng)
        items = []
        if rng.random() < 0.7:  # quiet phrase
            at = rng.uniform(0.2, max(0.3, P - 1.6))
            ln = rng.uniform(0.5, 1.4) if rng.random() < 0.6 else rng.uniform(2, 8)
            at = min(at, max(0.2, P - ln - 4))
            items.append(("s", at, ln, floor + rng.uniform(10, 26)))
        for _ in range(rng.integers(0, 3)):  # quiet short words
            at = rng.uniform(0.2, P - 0.6)
            items.append(("s", at, rng.uniform(0.12, 0.45), floor + rng.uniform(10, 22)))
        for _ in range(rng.integers(0, 5)):  # breaths
            at = rng.uniform(0.2, P - 0.6)
            items.append(("n", at, rng.uniform(0.08, 0.45), floor + rng.uniform(7, 17)))
        if STRADDLE and items and items[0][0] == "s":  # a train/sound starting near the phrase's 3 s reach
            _k, pat, pln, plv = items[0]
            el = rng.uniform(0.04, 0.3)
            gap = rng.uniform(0.1, 0.39)
            n = int(rng.integers(1, 8))
            at = pat + pln + rng.uniform(2.3, 3.1)
            lvl = floor + rng.uniform(8, 22)
            for k in range(n):
                a = at + k * (el + gap)
                if a + el < P - 0.1:
                    items.append(("n", a, el, lvl))
            if rng.random() < 0.5:  # quieter speech after it
                a = at + n * (el + gap) + rng.uniform(0.1, 1.5)
                if a + 0.3 < P - 0.1:
                    items.append(("s", a, rng.uniform(0.3, 1.2), plv - rng.uniform(6, 16)))
        for _ in range(rng.integers(0, 3)):  # a train of clicks / steps / short breaths
            el = rng.uniform(0.04, 0.2)
            gap = rng.uniform(0.15, 0.39) if not DENSE else rng.uniform(0.1, 0.3)
            n = int(rng.integers(2, 10))
            at = rng.uniform(0.2, P - 0.5)
            lvl = floor + rng.uniform(8, 22)
            for k in range(n):
                a = at + k * (el + gap)
                if a + el < P - 0.1:
                    items.append(("n", a, el, lvl))
        for kind, at, ln, lvl in items:
            ln = min(ln, P - at - 0.05)
            if ln <= 0:
                continue
            lib.add(buf, at, ln, lvl, rng)
            (speech if kind == "s" else nonspeech).append((t + at, t + at + ln))
        emit(buf)
    return np.concatenate(parts), speech, nonspeech


def measure(tag, wav, speech, nonspeech):
    pl = lib.plan(tag, wav)
    rg = pl.ranges
    sp = [(int(a * SR), int(b * SR)) for a, b in speech]
    got = sum(lib.covered(rg, a, b) for a, b in sp)
    lost = [(round(a / SR, 2), round(b / SR, 2), round((b - a - lib.covered(rg, a, b)) / SR, 2)) for a, b in sp if lib.covered(rg, a, b) < b - a]
    cuts = [r[1] for r, s in zip(rg, rg[1:]) if r[1] == s[0]]
    cuts_in = [round(c / SR, 2) for c in cuts if any(a < c < b for a, b in sp)]
    edges = [w for w in pl.windows]
    win_in = 0
    for (ws, we), (rs, re) in zip(pl.windows, rg):
        for e in ((ws,) if ws != rs else ()) + ((we,) if we != re else ()):
            if any(a < e < b for a, b in sp):
                win_in += 1
    total_dec = sum(b - a for a, b in rg) / SR
    ps = lib.pauses(tag, wav)
    p_in_speech = sum(max(0, min(pb, b) - max(pa, a)) for a, b in speech for pa, pb in ps if pa < b and pb > a)
    empty = [r for r in rg if not any(lib.covered([r], a, b) for a, b in sp)]
    return dict(empty=len(empty), empty_s=sum(b - a for a, b in empty) / SR, got=got, lost=lost, cuts_in=cuts_in, win_in=win_in, dec=total_dec, n=len(rg), pin=p_in_speech, ranges=rg)


def main():
    global diffs
    rng = np.random.default_rng(SEED)
    agg = {"old": dict(got=0, cuts_in=0, win_in=0, dec=0.0, n=0, pin=0.0, empty=0, empty_s=0.0), "head": dict(got=0, cuts_in=0, win_in=0, dec=0.0, n=0, pin=0.0, empty=0, empty_s=0.0)}
    worse = []
    better = []
    diffs = 0
    for i in range(N):
        wav, speech, nonspeech = scene(rng)
        m = measure("old", wav, speech, nonspeech)
        p = measure("head", wav, speech, nonspeech)
        for tag, r in (("old", m), ("head", p)):
            a = agg[tag]
            a["got"] += r["got"]; a["cuts_in"] += len(r["cuts_in"]); a["win_in"] += r["win_in"]; a["dec"] += r["dec"]; a["n"] += r["n"]; a["pin"] += r["pin"]; a["empty"] += r["empty"]; a["empty_s"] += r["empty_s"]
        if m["ranges"] != p["ranges"]:
            diffs += 1
        flags = []
        if p["got"] < m["got"]:
            flags.append(f"speech lost {(m['got'] - p['got']) / SR:.2f}s main_lost={m['lost']} pr_lost={p['lost']}")
        if len(p["cuts_in"]) > len(m["cuts_in"]):
            flags.append(f"cuts in speech main {m['cuts_in']} pr {p['cuts_in']}")
        if p["win_in"] > m["win_in"]:
            flags.append(f"window edges in speech main {m['win_in']} pr {p['win_in']}")
        if p["pin"] > m["pin"] + 1e-6:
            flags.append(f"retime pause inside speech main {m['pin']:.2f}s pr {p['pin']:.2f}s")
        if p['empty'] > m['empty']:
            flags.append(f"speechless ranges main {m['empty']} pr {p['empty']}")
        if p['got'] > m['got']:
            better.append(i)
        if flags:
            worse.append((i, flags))
    print(f"min_silence {MS} scenes {N} seed {SEED}: plans differ in {diffs}")
    for tag in ("old", "head"):
        a = agg[tag]
        print(tag, f"speech decoded {a['got'] / SR:.1f}s cuts-in-speech {a['cuts_in']} window-edges-in-speech {a['win_in']} decoded {a['dec']:.1f}s ranges {a['n']} retime-pause-in-speech {a['pin']:.2f}s speechless-ranges {a['empty']} ({a['empty_s']:.1f}s)")
    print("scenes where PR keeps more speech:", len(better), " scenes where PR is worse:", len(worse))
    for i, f in worse[:20]:
        print(" scene", i, f)


if __name__ == "__main__":
    main()
