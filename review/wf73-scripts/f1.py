import sys
from types import SimpleNamespace
import numpy as np
import types
ort=types.ModuleType('onnxruntime'); ort.get_available_providers=lambda:['CPUExecutionProvider']
sys.modules.setdefault('onnx_asr', types.ModuleType('onnx_asr')); sys.modules.setdefault('onnxruntime', ort)
from parakeet_service import routes
from parakeet_service.config import TARGET_SR

def _samples(r): return [(int(s*TARGET_SR), int(e*TARGET_SR)) for s, e in r]
def _prepared(ranges_sec, windows_sec):
    ranges = _samples(ranges_sec); windows = _samples(windows_sec)
    return routes._PreparedAudio(waveform=None, ranges=ranges, windows=windows, speech=[],
        pieces=[np.zeros(e-s, dtype=np.float32) for s, e in windows], duration=max(e for _s, e in ranges_sec))
def _result(tokens, ts):
    return SimpleNamespace(text="".join(tokens).strip(), tokens=tokens, timestamps=ts)

def run(name, left, right, lw=(0.0, 25.0), rw=(15.0, 40.0)):
    # left/right: list of (word, absolute time)
    l = _result([" "+w for w, _ in left], [t - lw[0] for _, t in left])
    r = _result([" "+w for w, _ in right], [t - rw[0] for _, t in right])
    prep = _prepared([(0.0, 20.0), (20.0, 40.0)], [lw, rw])
    text, segs, words = routes._stitch(prep, [l, r])
    print(f"{name}: text={text!r}")
    print("   words:", [(w['word'], round(w['start'], 3), round(w['end'], 3)) for w in words])

run("dropped-right-only", [("A", 19.5), ("B", 20.4)], [("A", 19.5), ("X", 20.1), ("B", 20.4)])
run("dropped-left-only (mirror)", [("A", 19.7), ("Y", 19.9), ("B", 20.4)], [("A", 19.7), ("B", 20.4)])
run("mirror, no match after cut", [("A", 19.7), ("Y", 19.9)], [("A", 19.7)])
run("gonna vs going to", [("A", 19.5), ("gonna", 20.1), ("B", 20.5)], [("A", 19.5), ("going", 20.08), ("to", 20.2), ("B", 20.5)])
run("gonna vs going to, no A", [("gonna", 20.1), ("B", 20.5)], [("going", 20.08), ("to", 20.2), ("B", 20.5)])
