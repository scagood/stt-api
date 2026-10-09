"""Default settings. Loud turn (40 s, -20 dBFS), a pause at -55 dBFS room tone holding a quiet
1 s phrase at -40, then a train of short sounds (clicks/steps/breaths) starting inside the
phrase's 3 s reach, then silence, then another loud turn. How far past the reach does each
side keep sound, is the pause still cut out, and how much pause audio is decoded?"""
import itertools
import sys

import numpy as np

import lib

SR = lib.SR
rows = []
for el, gap, n, lvl, pause_len in itertools.product(
    (0.06, 0.08, 0.1, 0.15), (0.2, 0.3, 0.38), range(2, 12), (-44, -40), (8.0, 10.0, 12.0)
):
    rng = np.random.default_rng(1)
    first = lib.turn(40, -20, rng)
    pause = lib.noise(pause_len, -55, rng)
    lib.add(pause, 1.0, 1.0, -40, rng)  # phrase 1.0-2.0 s; reach ends at 5.0 s
    t0 = 4.8
    train = [(t0 + i * (el + gap), el) for i in range(n)]
    end_train = train[-1][0] + el
    if end_train > pause_len - 0.3:
        continue
    for at, s in train:
        lib.add(pause, at, s, lvl, rng)
    last = lib.turn(40, -20, rng)
    wav = np.concatenate([first, pause, last])
    off = 40.0
    out = {}
    for tag in ("main", "pr"):
        rg = lib.plan(tag, wav).ranges
        dec = lib.covered(rg, int(off * SR), int((off + pause_len) * SR)) / SR
        sg = lib.segs(tag, wav)
        inpause = [(a / SR - off, b / SR - off) for a, b in sg if b / SR > off + 0.1 and a / SR < off + pause_len - 0.1]
        lastkept = max((b for a, b in inpause), default=None)
        ps = [(a - off, b - off) for a, b in lib.pauses(tag, wav) if b > off and a < off + pause_len]
        out[tag] = (round(dec, 2), len(rg), lastkept and round(lastkept, 2), ps)
    rows.append(((el, gap, n, lvl, pause_len, round(end_train, 2)), out))

diff = [r for r in rows if r[1]["main"][:3] != r[1]["pr"][:3]]
print("cases", len(rows), "differ", len(diff))
more = sorted((r for r in rows if r[1]["pr"][0] > r[1]["main"][0] + 0.01), key=lambda r: r[1]["main"][0] - r[1]["pr"][0])
less = [r for r in rows if r[1]["pr"][0] < r[1]["main"][0] - 0.01]
print("PR decodes more pause audio:", len(more), " less:", len(less))
for k, o in more[:15]:
    print(k, "main dec %.2f n=%d lastseg %s | PR dec %.2f n=%d lastseg %s" % (o["main"][0], o["main"][1], o["main"][2], o["pr"][0], o["pr"][1], o["pr"][2]))
ext = max(((o["pr"][2] or 0) - 5.0, k) for k, o in rows)
print("max PR segment end past reach (5.0 s):", ext)
ext = max(((o["main"][2] or 0) - 5.0, k) for k, o in rows)
print("max main segment end past reach (5.0 s):", ext)
