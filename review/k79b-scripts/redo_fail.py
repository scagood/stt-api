import asyncio, sys, types, logging
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
ort = types.ModuleType("onnxruntime"); ort.get_available_providers = lambda: ["CPUExecutionProvider"]
sys.modules.setdefault("onnx_asr", types.ModuleType("onnx_asr")); sys.modules.setdefault("onnxruntime", ort)
import numpy as np
from parakeet_service import routes, chunker
from parakeet_service.config import TARGET_SR as SR
logging.disable(logging.CRITICAL)
from parakeet_service.model import ModelLoadError
rng = np.random.default_rng(0)
wav = (rng.standard_normal(30 * SR) * 0.1).astype(np.float32)  # loud throughout: VAD hears speech
prep = routes._PreparedAudio(waveform=wav, ranges=[(0, 30*SR)], windows=[(0, 30*SR)], speech=None, pieces=[wav], duration=30.0)
first = SimpleNamespace(text="Hello there. Then", tokens=[" Hello", " there", ".", " Then"], timestamps=[0.4, 0.8, 1.0, 9.0])
class W:
    async def submit_many(self, pieces, key):
        raise ModelLoadError("parakeet-v3:fp32 evicted and failed to reload")
req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(worker=W(), ready=True, audio_pool=None)))
try:
    res = asyncio.run(routes._redo_stalled(req, [prep], [first], "parakeet-v3:fp32"))
    print("returned", [r.text for r in res])
except Exception as exc:
    print("raised", type(exc).__name__, getattr(exc, "status_code", ""), exc)
