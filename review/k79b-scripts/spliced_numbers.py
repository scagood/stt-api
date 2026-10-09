import asyncio, sys, types, logging
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
ort = types.ModuleType("onnxruntime"); ort.get_available_providers = lambda: ["CPUExecutionProvider"]
sys.modules.setdefault("onnx_asr", types.ModuleType("onnx_asr")); sys.modules.setdefault("onnxruntime", ort)
import numpy as np
from parakeet_service import routes
from parakeet_service.config import TARGET_SR as SR
logging.disable(logging.CRITICAL)
wav = np.full(30 * SR, 0.1, dtype=np.float32)
prep = routes._PreparedAudio(waveform=wav, ranges=[(0, 30*SR)], windows=[(0, 30*SR)], speech=None, pieces=[wav], duration=30.0)
routes.speech_segments = lambda w: [(0, w.size)]
first = SimpleNamespace(text="Hello there. Then we left.", tokens=[" Hello", " there", ".", " Then", " we", " left", "."],
                        timestamps=[0.4, 0.8, 1.0, 9.0, 9.3, 9.6, 9.9])
# redo of 0-11 s (stretch 1.32-9.0 s with 2 s after): hears the skipped "It cost $5 in 1984,"
again = SimpleNamespace(text="", tokens=[" Hello", " there", ".", " It", " cost", " ", "$", "5", " in", " ", "1", "9", "8", "4", ",", " Then"],
                        timestamps=[0.4, 0.8, 1.0, 2.0, 2.4, 3.0, 3.0, 3.1, 4.0, 4.4, 4.4, 4.5, 4.6, 4.7, 5.0, 8.96])
class W:
    async def submit_many(self, pieces, key):
        print("redo windows", [(float(p[0]) if False else None, p.size / SR) for p in pieces]); return [again] + [SimpleNamespace(text="", tokens=[], timestamps=[]) for _ in pieces[1:]]
req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(worker=W(), ready=True, audio_pool=None)))
res = asyncio.run(routes._redo_stalled(req, [prep], [first], "parakeet-v3:fp32"))
for speak in (False, True):
    text, segs, words = routes._stitch(prep, res, speak=speak, language="en")
    print("speak", speak, "|", text, "|", [w["word"] for w in words], "|", [s["segment"] for s in segs])
    print("   text==words:", text.split() == [w["word"] for w in words], "starts sorted:", [round(w["start"],2) for w in words])
