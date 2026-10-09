"""Random check: _stitch keeps exactly what the documented rule says.

Reference: matched pairs (same DP as routes) go left if both < cut else right;
unmatched words between the last left pair and first right pair go by own time;
unmatched words before the last left pair: left keeps, right drops; after the
first right pair: left drops, right keeps. Words outside reach: own time.
"""
import random
import sys

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[1] + "/tests")
from types import SimpleNamespace  # noqa: E402

import conftest  # noqa: E402,F401
import numpy as np  # noqa: E402

from parakeet_service import routes  # noqa: E402
from parakeet_service.config import TARGET_SR  # noqa: E402


def pairs_of(left, right, cut, reach):
    lows = [(i, at, w) for i, (w, at) in enumerate(left) if abs(at - cut) <= reach]
    highs = [(j, at, w) for j, (w, at) in enumerate(right) if abs(at - cut) <= reach]
    steps = [[(0, 0.0, ())] * (len(highs) + 1) for _ in range(len(lows) + 1)]
    for a in reversed(range(len(lows))):
        for b in reversed(range(len(highs))):
            options = [steps[a + 1][b], steps[a][b + 1]]
            (i, at, key), (j, there, other) = lows[a], highs[b]
            if key == other and abs(at - there) <= routes._SEAM_MATCH_SEC:
                m, c, p = steps[a + 1][b + 1]
                options.append((m + 1, c - abs(at - there), ((i, j, at, there), *p)))
            steps[a][b] = max(options)
    return list(steps[0][0][2])


def prep(ranges, windows):
    rs = [(int(round(s * TARGET_SR)), int(round(e * TARGET_SR))) for s, e in ranges]
    ws = [(int(round(s * TARGET_SR)), int(round(e * TARGET_SR))) for s, e in windows]
    return routes._PreparedAudio(waveform=None, ranges=rs, windows=ws, speech=[],
                                 pieces=[np.zeros(e - s, dtype=np.float32) for s, e in ws], duration=60.0)


rng = random.Random(7)
VOC = ["a", "b", "c", "the", "no"]
bad = 0
for trial in range(20000):
    n = 3
    ranges = [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)]
    windows = [(0.0, 25.0), (15.0 + rng.choice([0, 0.03, 0.07]), 45.0), (35.0 + rng.choice([0, 0.03, 0.07]), 60.0)]
    if rng.random() < 0.15:
        k = rng.randrange(3)
        windows[k] = ranges[k]  # re-decoded without context
    pieces = []
    for p in range(n):
        ws, we = windows[p]
        t = ws + rng.uniform(0, 0.3)
        words = []
        while t < we:
            # dense only near cuts
            near = min(abs(t - 20), abs(t - 40)) < 1.5
            if near:
                tt = round(t, 3)
                if min(abs(tt - 20), abs(tt - 40)) < 1e-6:
                    tt += 0.001  # off the float boundary
                words.append((rng.choice(VOC), tt))
                t += rng.choice([0.08, 0.16, 0.24, 0.32])
            else:
                t += 0.9
        pieces.append(words)
    results = [SimpleNamespace(text=" ".join(w for w, _t in ws_), tokens=[" " + w for w, _t in ws_], timestamps=[t - windows[p][0] for _w, t in ws_])
               for p, ws_ in enumerate(pieces)]
    _text, _segs, out = routes._stitch(prep(ranges, windows), results)
    # reference
    keep = [[None] * len(pc) for pc in pieces]
    for p, pc in enumerate(pieces):
        s, e = ranges[p]
        tail = e if windows[p][1] > e else float("inf")
        head = s
        for i, (_w, t) in enumerate(pc):
            keep[p][i] = [head <= t, t < tail]  # [after_start, before_end]
    for idx in range(n - 1):
        cut = ranges[idx][1]
        if windows[idx][1] == cut == windows[idx + 1][0]:
            continue
        reach = min(1.0, (cut - ranges[idx][0]) / 2, (ranges[idx + 1][1] - cut) / 2)
        pr = pairs_of(pieces[idx], pieces[idx + 1], cut, reach)
        if not pr:
            continue
        lefts = [q for q in pr if max(q[2], q[3]) < cut]
        rights = [q for q in pr if max(q[2], q[3]) >= cut]
        li = lefts[-1][0] if lefts else -1
        lj = lefts[-1][1] if lefts else -1
        ri = rights[0][0] if rights else 10 ** 9
        rj = rights[0][1] if rights else 10 ** 9
        for i in range(len(pieces[idx])):
            if i <= li:
                keep[idx][i][1] = True
            elif i >= ri:
                keep[idx][i][1] = False
        for j in range(len(pieces[idx + 1])):
            if j <= lj:
                keep[idx + 1][j][0] = False
            elif j >= rj:
                keep[idx + 1][j][0] = True
    expected = []
    for p, pc in enumerate(pieces):
        if not pc:
            continue
        expected += [w for i, (w, _t) in enumerate(pc) if all(keep[p][i])]
    got = [w["word"] for w in out]
    if got != expected:
        bad += 1
        if bad <= 5:
            print("MISMATCH", windows, pieces, got, expected)
print("trials=20000 mismatches", bad)
