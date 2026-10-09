"""Bounds fuzz with cheap word-like noise: piece counts, sizes, windows,
contiguity, smallest part, lost audio (outer margin vs elsewhere)."""
import random, zlib
from collections import Counter
from h import *
from fuzz80 import layout


def cheap(rng, segs, total):
    env = np.full(total, db(-65), np.float32)
    for a, b in segs:
        t = a
        while t < b:
            w = int(rng.uniform(0.12, 0.6) * SR); g = int(rng.uniform(0.0, 0.35) * SR)
            env[t:min(b, t + w)] = db(rng.uniform(-26, -14))
            env[min(b, t + w):min(b, t + w + g)] = db(rng.uniform(-50, -28))
            t += w + g
    return (np.random.default_rng(rng.integers(1 << 30)).standard_normal(total).astype(np.float32) * env)


cases = [("v2 ctx5", 25, 30, 5, 3), ("v2 ctx7.5", 25, 30, 7.5, 3), ("v3 ctx5", 60, 75, 5, 3), ("whisper", 25, 30, 0, 3),
         ("v2 ctx5 trim1", 25, 30, 5, 1), ("v2 ctx5 trim0.5", 25, 30, 5, 0.5), ("v2 ctx5 trim6", 25, 30, 5, 6),
         ("short 8/10 ctx2.5", 8, 10, 2.5, 3), ("whisper trim8", 25, 30, 0, 8)]
N = int(sys.argv[1]); scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
for name, target, mx, ctx, trim in cases:
    for m in M.values(): m.CHUNK_TRIM_SILENCE_SEC = trim
    own = int((mx - 2 * ctx) * SR)
    c = Counter(); ex = {}
    rng = random.Random(f"b|{name}|{scale}")
    for i in range(N):
        segs_s, total_s = layout(rng, scale)
        if total_s <= mx: continue
        segs = [(int(a * SR), int(b * SR)) for a, b in segs_s]; total = int(total_s * SR)
        wav = cheap(np.random.default_rng(zlib.crc32(f"{name}|{scale}|{i}".encode())), segs, total)
        P = {}
        for k, m in M.items():
            m._speech_segments = lambda _w, s=segs: list(s)
            p = m.plan_chunks(wav, target_sec=target, max_sec=mx, min_sec=20.0, context_sec=ctx)
            P[k] = (p.ranges, p.windows)
        (rm, wm), (rp, wp) = P["main"], P["pr"]
        c["layouts"] += 1
        if len(rp) != len(rm): c["diff_pieces"] += 1; ex.setdefault("pieces", (segs_s, total_s, sec(rm), sec(rp)))
        if any(b - a > own for a, b in rp) and total > int(mx * SR): c["range>own"] += 1; ex.setdefault("own", (segs_s, total_s, sec(rp)))
        if any(b - a > int(mx * SR) for a, b in wp): c["window>max"] += 1; ex.setdefault("win", (segs_s, total_s, sec(wp)))
        if any(not (wa <= a and b <= wb) for (a, b), (wa, wb) in zip(rp, wp)): c["window!>=range"] += 1
        if any(b > cc for (a, b), (cc, d) in zip(rp, rp[1:])): c["overlap"] += 1
        meet_m = sum(b == cc for (a, b), (cc, d) in zip(rm, rm[1:])); meet_p = sum(b == cc for (a, b), (cc, d) in zip(rp, rp[1:]))
        if meet_m != meet_p: c["contiguity_changed"] += 1
        # smallest new part vs own/4, only parts the PR made shorter than main's shortest
        mn_m = min(b - a for a, b in rm); mn_p = min(b - a for a, b in rp)
        if mn_p < mn_m: c["shorter_min_part"] += 1
        if mn_p < own / 4 and mn_p < mn_m: c["part<own/4"] += 1; ex.setdefault("short", (segs_s, total_s, sec(rm), sec(rp)))
        gone = covered(wm, total) & ~covered(wp, total)
        if gone.any():
            idx = np.flatnonzero(gone)
            inner = ((idx >= segs[0][0]) & (idx < segs[-1][1])).any()
            c["lost_outer_margin"] += 1
            if inner: c["lost_inner"] += 1; ex.setdefault("inner", (segs_s, total_s, sec(wm), sec(wp)))
        gr = covered(rm, total) & ~covered(rp, total)
        if gr.any() and not gone.any(): c["range_only_lost"] += 1
    print(f"{name:18s} scale {scale}: {dict(c)}")
    for k, v in ex.items(): print("   ex", k, v)
