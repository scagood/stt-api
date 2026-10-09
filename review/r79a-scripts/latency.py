"""Latency of _redo_stalled for a one-piece clip with nothing to redo, and with
a 3.5 s pause (VAD runs, nothing redone). Pool: a ThreadPoolExecutor whose
submit is counted.

python latency.py <worktree>"""
import asyncio
import concurrent.futures
import statistics
import sys
import time
import types
from types import SimpleNamespace

sys.path.insert(0, sys.argv[1])
ort = types.ModuleType("onnxruntime")
ort.get_available_providers = lambda: ["CPUExecutionProvider"]
sys.modules.setdefault("onnx_asr", types.ModuleType("onnx_asr"))
sys.modules.setdefault("onnxruntime", ort)
import numpy as np  # noqa: E402

from parakeet_service import routes  # noqa: E402
from parakeet_service.config import TARGET_SR as SR  # noqa: E402


class CountingPool(concurrent.futures.ThreadPoolExecutor):
    submits = 0

    def submit(self, *a, **k):
        CountingPool.submits += 1
        return super().submit(*a, **k)


vad_calls = []
if hasattr(routes, "speech_segments"):
    real_vad = routes.speech_segments

    def counted(wav):
        vad_calls.append(wav.size)
        return real_vad(wav)

    routes.speech_segments = counted


class Worker:
    def __init__(self):
        self.calls = 0

    async def submit_many(self, pieces, key):
        self.calls += len(pieces)
        return [SimpleNamespace(text="", tokens=[], timestamps=[]) for _ in pieces]


def clip(seconds, pause_at=None, pause=3.5):
    rng = np.random.default_rng(0)
    wav = (rng.standard_normal(int(seconds * SR)) * 0.1).astype(np.float32)
    toks, ts = [], []
    t = 0.3
    i = 0
    while t < seconds - 0.3:
        if pause_at is not None and pause_at <= t < pause_at + pause:
            wav[int(t * SR): int((pause_at + pause) * SR)] = 0
            t = pause_at + pause + 0.1
            continue
        toks.append(f" w{i}")
        ts.append(round(t, 2))
        t += 0.4
        i += 1
    prep = routes._PreparedAudio(waveform=wav, ranges=[(0, wav.size)], windows=[(0, wav.size)], speech=None,
                                 pieces=[wav], duration=seconds)
    return prep, SimpleNamespace(text=" ".join(x.strip() for x in toks), tokens=toks, timestamps=ts)


async def measure(prep, res, pool, n=200):
    worker = Worker()
    req = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(worker=worker, ready=True, audio_pool=pool)))
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        await routes._redo_stalled(req, [prep], [res], "parakeet-v3:fp32")
        times.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(times), max(times), worker.calls


pool = CountingPool(max_workers=2)
for name, (prep, res) in [("75 s clip, words every 0.4 s, nothing to redo", clip(75)),
                          ("75 s clip with a 3.5 s silent pause", clip(75, pause_at=30.0))]:
    CountingPool.submits = 0
    vad_calls.clear()
    med, mx, decodes = asyncio.run(measure(prep, res, pool))
    print(f"{name}: median {med:.0f} us, max {mx:.0f} us per call; pool submits/call={CountingPool.submits / 200:.2f} "
          f"VAD calls/call={len(vad_calls) / 200:.2f} decodes={decodes}")
pool.shutdown()
