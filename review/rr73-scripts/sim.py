"""Usage: python sim.py <code dir> [mishear] [drop] [n]. Random cuts in speech -> lost/dup counts."""
import random
import sys

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[1] + "/tests")
from types import SimpleNamespace  # noqa: E402

import conftest  # noqa: E402,F401
import numpy as np  # noqa: E402

from parakeet_service import routes  # noqa: E402
from parakeet_service.config import TARGET_SR  # noqa: E402

MIS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
DROP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
N = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
REP = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
rng = random.Random(1234)
GRID = 0.08


def prep(ranges, windows):
    rs = [(int(round(s * TARGET_SR)), int(round(e * TARGET_SR))) for s, e in ranges]
    ws = [(int(round(s * TARGET_SR)), int(round(e * TARGET_SR))) for s, e in windows]
    return routes._PreparedAudio(waveform=None, ranges=rs, windows=ws, speech=[],
                                 pieces=[np.zeros(e - s, dtype=np.float32) for s, e in ws], duration=40.0)


lost = dup = cuts_bad = 0
for trial in range(N):
    t = 18.0 + rng.uniform(0, 0.5)
    truth = []
    k = 0
    while t < 22.0:
        name = f"w{k}"
        if REP and truth and rng.random() < REP:
            name = truth[-1][0]  # a word said again
        truth.append((name, t, k))
        k += 1
        t += rng.uniform(0.12, 0.5)
    wins = [(0.0, 25.0), (15.0 + rng.uniform(0, GRID), 40.0)]
    heard = []
    for p, (ws, _we) in enumerate(wins):
        out = []
        for name, tt, kk in truth:
            if DROP and rng.random() < DROP:
                continue
            word = name
            if rng.random() < MIS:
                word = f"x{kk}_{p}"
            rel = tt - ws + rng.gauss(0.02, 0.025)
            rel = max(0.0, round(rel / GRID) * GRID)
            out.append((word, kk, rel))
        out.sort(key=lambda x: x[2])
        heard.append(out)
    results = [SimpleNamespace(text=" ".join(w for w, _k, _t in h), tokens=[" " + w for w, _k, _t in h],
                               timestamps=[tt for _w, _k, tt in h]) for h in heard]
    # remember source word ids by (piece, position)
    _text, _segs, words = routes._stitch(prep([(0.0, 20.0), (20.0, 40.0)], wins), results)
    # map output words back: each piece's kept words are in order; recover by name+approx time
    ids = {}
    for p, h in enumerate(heard):
        for w, kk, tt in h:
            ids.setdefault((w, round(wins[p][0] + tt, 3)), []).append(kk)
    counts = {}
    for w in words:
        # word start is clamped to the range; find the source by name, nearest time
        cands = [(abs(at - w["start"]), kk) for (name, at), kks in ids.items() if name == w["word"] for kk in kks]
        _d, kk = min(cands)
        counts[kk] = counts.get(kk, 0) + 1
    bad = False
    for name, tt, kk in truth:
        c = counts.get(kk, 0)
        if c == 0:
            lost += 1
            bad = True
        elif c > 1:
            dup += c - 1
            bad = True
    cuts_bad += bad
print(f"mis={MIS} drop={DROP} rep={REP} N={N}: lost={lost} dup={dup} cuts_with_error={cuts_bad}")
