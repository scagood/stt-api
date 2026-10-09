import asyncio, concurrent.futures, sys, types
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
ort = types.ModuleType("onnxruntime"); ort.get_available_providers = lambda: ["CPUExecutionProvider"]
sys.modules.setdefault("onnx_asr", types.ModuleType("onnx_asr")); sys.modules.setdefault("onnxruntime", ort)
import numpy as np
from parakeet_service import routes
from parakeet_service.config import TARGET_SR as SR
class Pool(concurrent.futures.ThreadPoolExecutor):
    n = 0
    def submit(self, *a, **k):
        Pool.n += 1
        return super().submit(*a, **k)
files, results = [], []
for k in range(3):
    wav = np.zeros(30 * SR, dtype=np.float32)
    files.append(routes._PreparedAudio(waveform=wav, ranges=[(0, wav.size)], windows=[(0, wav.size)],
                 speech=None if hasattr(routes, "speech_segments") else [], pieces=[wav], duration=30.0))
    results.append(SimpleNamespace(text="", tokens=[f" w{i}" for i in range(70)], timestamps=[0.4 * i for i in range(70)]))
pool = Pool(1)
req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(worker=None, ready=True, audio_pool=pool)))
asyncio.run(routes._redo_stalled(req, files, results, "parakeet-v3:fp32"))
print("batch of 3 one-piece files, nothing to redo: pool submits =", Pool.n)
asyncio.run(routes._redo_stalled(req, files[:1], results[:1], "parakeet-v3:fp32"))
print("then one file alone: pool submits total =", Pool.n)
