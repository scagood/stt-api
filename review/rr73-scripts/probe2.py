import sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[1] + "/tests")
import conftest  # noqa
exec(open(sys.argv[0].replace("probe2.py", "probe.py")).read().split('R2 = [')[0].split('from parakeet_service import routes')[1].join(['from parakeet_service import routes', '']) if False else "")
from types import SimpleNamespace
import numpy as np
from parakeet_service import routes
from parakeet_service.config import TARGET_SR
def prep(ranges, windows):
    S = lambda r: [(int(round(s * TARGET_SR)), int(round(e * TARGET_SR))) for s, e in r]
    rs, ws = S(ranges), S(windows)
    return routes._PreparedAudio(waveform=None, ranges=rs, windows=ws, speech=[], pieces=[np.zeros(e - s, dtype=np.float32) for s, e in ws], duration=40.0)
def run(name, pieces, windows):
    res = [SimpleNamespace(text=" ".join(w for w, _ in p), tokens=[" " + w for w, _ in p], timestamps=[t - win[0] for _, t in p]) for p, win in zip(pieces, windows)]
    _t, _s, words = routes._stitch(prep([(0.0, 20.0), (20.0, 40.0)], windows), res)
    print(f"{name:38s} -> " + " ".join(f"{w['word']}@{w['start']:.2f}-{w['end']:.2f}" for w in words))
W = [(0.0, 25.0), (15.03, 40.0)]
run("B straddle (R earlier), W right-only", [[("A", 19.5), ("B", 20.05), ("C", 20.5)], [("A", 19.5), ("B", 19.87), ("W", 19.95), ("C", 20.5)]], W)
run("B straddle (L earlier), Z left-only", [[("A", 19.5), ("B", 19.87), ("Z", 19.95), ("C", 20.5)], [("A", 19.5), ("B", 20.03), ("C", 20.5)]], W)
