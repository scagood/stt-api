"""Worst case: one VAD segment over the whole file (fixed gate under a noise
bed), many forced cuts. Times plan_chunks main vs PR; peak RSS delta."""
import time, resource, tracemalloc
from h import *
hours = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
src = sys.argv[2] if len(sys.argv) > 2 else None
if src:
    wav = np.fromfile(src, dtype=np.float32)
    reps = int(np.ceil(hours * 3600 * SR / wav.size))
    wav = np.tile(wav, reps)[: int(hours * 3600 * SR)]
else:
    rng = np.random.default_rng(0)
    one = synth(rng, [(0, 600 * SR)], 600 * SR)[0]
    wav = np.tile(one, int(np.ceil(hours * 6)))[: int(hours * 3600 * SR)]
total = wav.size
segs = [(0, total)]
for cname, cfg in CONFIGS.items():
    for k, m in M.items():
        tracemalloc.start()
        t = time.perf_counter()
        r, w = plan(m, wav, segs, cfg)
        dt = time.perf_counter() - t
        cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        print(f"{cname:8s} {k:5s} {hours}h: {len(r)} pieces, {dt:.2f}s, peak traced {peak/1e6:.1f} MB")
