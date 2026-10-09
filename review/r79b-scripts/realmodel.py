import os, numpy as np, onnx_asr, time
S = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad"
_M = []
def model():
    if not _M:
        m = onnx_asr.load_model("nemo-conformer-tdt", f"{S}/r79b-model", quantization="int8", providers=["CPUExecutionProvider"])
        _M.append(m.with_timestamps())
    return _M[0]
def audio(name):
    return np.fromfile(f"{S}/r79b-audio/{name}.f32", dtype=np.float32)
def recognize(wavs):
    r = model().recognize(wavs)
    return r
