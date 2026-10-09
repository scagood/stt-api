import sys
from types import SimpleNamespace
import numpy as np
import types
try:
    import onnxruntime
except ImportError:
    o=types.ModuleType('onnxruntime'); o.get_available_providers=lambda:['CPUExecutionProvider']; sys.modules['onnxruntime']=o; sys.modules['onnx_asr']=types.ModuleType('onnx_asr')
from parakeet_service import routes
from parakeet_service.config import TARGET_SR

def _prepared(ranges_sec, windows_sec):
    ranges = [(int(s*TARGET_SR), int(e*TARGET_SR)) for s, e in ranges_sec]
    windows = [(int(s*TARGET_SR), int(e*TARGET_SR)) for s, e in windows_sec]
    return routes._PreparedAudio(waveform=None, ranges=ranges, windows=windows, speech=[],
        pieces=[np.zeros(e-s, dtype=np.float32) for s, e in windows], duration=max(e for _s, e in ranges_sec))

def _result(words):
    tokens = [" "+w for w, _ in words]; ts = [t for _, t in words]
    return SimpleNamespace(text="".join(tokens).strip(), tokens=tokens, timestamps=ts)

def run(name, ranges, windows, pieces):
    # pieces: list of word lists with ABSOLUTE times; convert to window-relative
    results = [_result([(w, round(t - win[0], 4)) for w, t in p]) for p, win in zip(pieces, windows)]
    text, segs, words = routes._stitch(_prepared(ranges, windows), results)
    print(f"{name}: {text!r} | " + " ".join(f"{w['word']}@{w['start']:.2f}-{w['end']:.2f}" for w in words) + " | segs=" + repr([s['segment'] for s in segs]))

R2 = [(0.0, 20.0), (20.0, 40.0)]
W2 = [(0.0, 25.0), (15.0, 40.0)]
run("F1a X", R2, W2, [[("A",19.5),("B",20.4)], [("A",19.5),("X",20.1),("B",20.4)]])
run("F1b Y+B", R2, W2, [[("A",19.7),("Y",19.9),("B",20.4)], [("A",19.7),("B",20.4)]])
run("F1c Y", R2, W2, [[("A",19.7),("Y",19.9)], [("A",19.7)]])
run("F1d gonna", R2, W2, [[("gonna",20.1),("B",20.5)], [("going",20.08),("to",20.2),("B",20.5)]])
run("F1d' gonna closer match before", R2, W2, [[("A",19.9),("gonna",20.1),("B",20.5)], [("A",19.9),("going",20.08),("to",20.2),("B",20.5)]])
run("straddle three L<cut", R2, W2, [[("A",19.5),("three",19.92),("B",20.5)], [("A",19.5),("free",20.03),("B",20.5)]])
run("straddle three L>cut", R2, W2, [[("A",19.5),("three",20.03),("B",20.5)], [("A",19.5),("free",19.92),("B",20.5)]])
W2b = [(0.0, 25.0), (15.03, 40.0)]
run("NIT Z", R2, W2b, [[("A",19.5),("B",19.87),("Z",19.95),("C",20.5)], [("A",19.5),("B",20.03),("C",20.5)]])
run("NIT W mirror", R2, W2b, [[("A",19.5),("B",20.05),("C",20.5)], [("A",19.5),("B",19.87),("W",19.95),("C",20.5)]])
