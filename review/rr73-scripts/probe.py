"""Usage: python probe.py <code dir>. Runs seam probes through _stitch."""
import sys

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[1] + "/tests")
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

import conftest  # noqa: E402,F401  (stubs onnxruntime etc.)
from parakeet_service import routes  # noqa: E402
from parakeet_service.config import TARGET_SR  # noqa: E402


def S(r):
    return [(int(round(s * TARGET_SR)), int(round(e * TARGET_SR))) for s, e in r]


def prep(ranges, windows):
    rs, ws = S(ranges), S(windows)
    return routes._PreparedAudio(
        waveform=None, ranges=rs, windows=ws, speech=[],
        pieces=[np.zeros(e - s, dtype=np.float32) for s, e in ws], duration=max(e for _s, e in ranges),
    )


def res(words):
    # words: list of (word, absolute time); window start subtracted by caller
    toks = [" " + w for w, _ in words]
    ts = [t for _, t in words]
    return SimpleNamespace(text="".join(toks).strip(), tokens=toks, timestamps=ts)


def run(name, pieces, ranges, windows):
    # pieces: list of [(word, absolute time)] per piece
    results = [res([(w, t - win[0]) for w, t in p]) for p, win in zip(pieces, windows)]
    text, segs, words = routes._stitch(prep(ranges, windows), results)
    print(f"{name:40s} -> {' '.join((w['word'] + '@' + format(w['start'], '.2f')) for w in words)}")


R2 = [(0.0, 20.0), (20.0, 40.0)]
W2 = [(0.0, 25.0), (15.0, 40.0)]

print("== F1 repros ==")
run("X right-only", [[("A", 19.5), ("B", 20.4)], [("A", 19.5), ("X", 20.1), ("B", 20.4)]], R2, W2)
run("Y left-only, match after", [[("A", 19.7), ("Y", 19.9), ("B", 20.4)], [("A", 19.7), ("B", 20.4)]], R2, W2)
run("Y left-only, no match after", [[("A", 19.7), ("Y", 19.9)], [("A", 19.7)]], R2, W2)
run("gonna/going to, B after", [[("gonna", 20.1), ("B", 20.5)], [("going", 20.08), ("to", 20.2), ("B", 20.5)]], R2, W2)
run("gonna/going to, A before closer", [[("A", 19.95), ("gonna", 20.1), ("B", 20.5)], [("A", 19.95), ("going", 20.08), ("to", 20.2), ("B", 20.5)]], R2, W2)
print("== correction: straddling word heard differently ==")
run("three@19.92 L / free@20.03 R", [[("one", 19.2), ("three", 19.92), ("four", 20.8)], [("one", 19.2), ("free", 20.03), ("four", 20.8)]], R2, W2)
run("three@20.03 L / free@19.92 R", [[("one", 19.2), ("three", 20.03), ("four", 20.8)], [("one", 19.2), ("free", 19.92), ("four", 20.8)]], R2, W2)
run("three@19.92 L / free@19.96 R", [[("one", 19.2), ("three", 19.92), ("four", 20.8)], [("one", 19.2), ("free", 19.96), ("four", 20.8)]], R2, W2)
run("three@20.03 L / free@20.07 R", [[("one", 19.2), ("three", 20.03), ("four", 20.8)], [("one", 19.2), ("free", 20.07), ("four", 20.8)]], R2, W2)

print("== repeated words ==")
for off in (15.03, 15.07):
    W = [(0.0, 25.0), (off, 40.0)]
    for lt in (19.92, 20.0):
        # "no no no" spanning the cut, left grid from 0, right from off
        L = [("no", lt - 0.4), ("no", lt), ("no", lt + 0.4)]
        Rr = [("no", round(off + round((t - off) / 0.08) * 0.08, 2)) for _w, t in L]
        run(f"no no no off={off} lt={lt}", [L, Rr], R2, W)
    # "the the" where right hears one
    run(f"the the / the off={off}", [[("of", 19.6), ("the", 19.92), ("the", 20.16), ("cat", 20.5)], [("of", 19.6), ("the", 20.03), ("cat", 20.5)]], R2, W)
    run(f"the / the the off={off}", [[("of", 19.6), ("the", 19.92), ("cat", 20.5)], [("of", 19.6), ("the", 19.85), ("the", 20.07), ("cat", 20.5)]], R2, W)
    run(f"I I I off={off}", [[("it", 19.44), ("I", 19.84), ("I", 20.08), ("I", 20.4)], [("it", 19.4 + off - 15.0), ("I", 19.8 + off - 15.0 + 0.0), ("I", 20.04 + off - 15.0), ("I", 20.44 + off - 15.0)]], R2, W)
    run(f"I I I L / I I R off={off}", [[("it", 19.44), ("I", 19.84), ("I", 20.08), ("I", 20.4)], [("it", 19.47), ("I", 19.87), ("I", 20.47)]], R2, W)

print("== 3 chunks ==")
R3 = [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)]
W3 = [(0.0, 25.0), (15.03, 45.0), (35.07, 60.0)]
run("3c test case", [[("one", 19.2), ("two", 20.0), ("three", 20.8)],
                    [("one", 19.19), ("two", 19.99), ("three", 20.79), ("four", 39.91), ("five", 40.71)],
                    [("four", 40.03), ("five", 40.75)]], R3, W3)
run("3c one-sided both cuts", [[("a", 19.5), ("Y", 19.9), ("b", 20.4)],
                              [("a", 19.5), ("b", 20.4), ("c", 39.5), ("X", 40.1), ("d", 40.4)],
                              [("c", 39.5), ("d", 40.4)]], R3, W3)
run("3c X at 2nd cut right-only", [[("a", 19.5), ("b", 20.4)],
                              [("a", 19.5), ("b", 20.4), ("c", 39.5), ("d", 40.4)],
                              [("c", 39.5), ("X", 40.1), ("d", 40.4)]], R3, W3)
# short middle piece (2 s): seams reach 1 s each
R3s = [(0.0, 20.0), (20.0, 22.0), (22.0, 40.0)]
W3s = [(0.0, 25.0), (15.0, 27.0), (17.0, 40.0)]
run("3c short middle", [[("a", 19.5), ("b", 19.96), ("c", 20.5), ("d", 21.0)],
                       [("a", 19.5), ("b", 20.02), ("c", 20.5), ("d", 21.0), ("e", 21.5), ("f", 21.98), ("g", 22.5)],
                       [("d", 21.0), ("e", 21.5), ("f", 22.03), ("g", 22.5)]], R3s, W3s)
run("3c short middle, mid word", [[("a", 19.5), ("b", 20.5), ("c", 21.0), ("d", 21.5)],
                       [("a", 19.5), ("b", 20.5), ("c", 21.0), ("d", 21.5)],
                       [("b", 20.5), ("c", 21.0), ("d", 21.5)]], R3s, W3s)
print("== neighbour re-decoded without context ==")
run("left redone", [[("one", 19.2), ("two", 19.92)], [("one", 19.23), ("two", 20.03), ("three", 20.83)]], R2, [(0.0, 20.0), (15.07, 40.0)])
run("left redone, Y right-only", [[("one", 19.2), ("two", 20.4)], [("one", 19.23), ("Y", 19.95), ("two", 20.43)]], R2, [(0.0, 20.0), (15.07, 40.0)])
run("right redone", [[("one", 19.2), ("two", 19.92), ("three", 20.8)], [("two", 20.0), ("three", 20.8)]], R2, [(0.0, 25.0), (20.0, 40.0)])
run("right redone, Z left-only after", [[("one", 19.2), ("Z", 20.1), ("three", 20.8)], [("three", 20.8)]], R2, [(0.0, 25.0), (20.0, 40.0)])
print("== straddle match then left-only before cut ==")
W = [(0.0, 25.0), (15.03, 40.0)]
run("B straddle, Z left-only <cut", [[("A", 19.5), ("B", 19.87), ("Z", 19.95), ("C", 20.5)], [("A", 19.5), ("B", 20.03), ("C", 20.5)]], R2, W)
run("Z left-only <cut before straddle B", [[("A", 19.5), ("Z", 19.79), ("B", 19.87), ("C", 20.5)], [("A", 19.5), ("B", 20.03), ("C", 20.5)]], R2, W)
