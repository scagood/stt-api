"""plan_chunks time on 2 h of speech-like audio, main vs #80."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from synth import MAIN, PR, CONFIGS, SR, synth, forced_cuts

hours = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
total = int(hours * 3600 * SR)
nrng = np.random.default_rng(7)
# phrases 2-15 s with 0.5-1.5 s pauses (the real VAD finds them), words with 20 dB gaps
segs, t = [], int(0.5 * SR)
while t < total - 20 * SR:
    L = int(nrng.uniform(2, 15) * SR)
    segs.append((t, t + L)); t += L + int(nrng.uniform(0.5, 1.5) * SR)
t0 = time.time()
wav, words = synth(segs, total, nrng, 20.0, (50, 350))
print(f"synth {hours} h: {time.time()-t0:.1f}s, {len(segs)} phrases", flush=True)
orig = {m: m._speech_segments for m in (MAIN, PR)}
for cname in ("v2 ctx5", "v3 ctx5", "whisper"):
    cfg = CONFIGS[cname]
    for label, seg_fn in (("real volume VAD", None), ("one segment (VAD patched)", lambda _w: [(0, total)])):
        res = {}
        for name, mod in (("main", MAIN), ("#80", PR)):
            mod._speech_segments = seg_fn or orig[mod]
            best = 1e9
            for _ in range(2):
                t0 = time.time()
                p = mod.plan_chunks(wav, target_sec=cfg["target"], max_sec=cfg["mx"], min_sec=20.0, context_sec=cfg["ctx"])
                best = min(best, time.time() - t0)
            res[name] = (best, len(p.ranges), len(forced_cuts(p.ranges, p.speech)))
        print(f"{cname:8s} {label:28s} main {res['main'][0]:.2f}s ({res['main'][1]} pieces, {res['main'][2]} forced) | "
              f"#80 {res['#80'][0]:.2f}s ({res['#80'][1]} pieces, {res['#80'][2]} forced)", flush=True)
