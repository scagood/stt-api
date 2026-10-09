"""A quiet word VAD misses in the trailing margin: main decodes it, #80 can drop it.
Real volume VAD (no patching); parakeet-v2 bounds (25/30, ctx 5)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from synth import MAIN, PR, SR, synth

found = 0
tried = 0
for seed in range(200):
    nrng = np.random.default_rng(seed)
    speech_len = nrng.uniform(36.0, 39.5)          # one run of speech, no pause >= 400 ms
    lead = 3.0
    total_s = lead + speech_len + 0.3
    total = int(total_s * SR)
    seg = (int(lead * SR), int((lead + speech_len) * SR))
    wav, words = synth([seg], total, nrng, 20.0, (50, 350))
    # a quiet word ("yes."), 0.3 s, 18 dB under the speech, 1.2-1.9 s before it
    at = lead - 0.3 - nrng.uniform(1.2, 1.9)
    a, b = int(at * SR), int((at + 0.3) * SR)
    wav[a:b] = nrng.standard_normal(b - a).astype(np.float32) * 10 ** ((-20 - 18) / 20)
    res = {}
    for name, mod in (("main", MAIN), ("#80", PR)):
        p = mod.plan_chunks(wav, target_sec=25.0, max_sec=30.0, min_sec=20.0, context_sec=5.0)
        heard = any(s < b and e > a for s, e in p.speech)
        covered = any(r0 <= a and b <= r1 for r0, r1 in p.ranges)
        in_window = any(w0 <= a and b <= w1 for w0, w1 in p.windows)
        res[name] = (heard, covered, in_window, [(round(r0 / SR, 2), round(r1 / SR, 2)) for r0, r1 in p.ranges])
    tried += 1
    if res["main"][1] and not res["#80"][1]:
        found += 1
        if found <= 2:
            print(f"seed {seed}: speech {lead:.2f}-{lead+speech_len:.2f}s, VAD speech {[(round(s/SR,2), round(e/SR,2)) for s,e in p.speech]}, quiet word {at:.2f}-{at+0.3:.2f}s (VAD heard it: {res['main'][0]})")
            print(f"   main ranges {res['main'][3]}  (word in a range: {res['main'][1]})")
            print(f"   #80  ranges {res['#80'][3]}  (word in a range: {res['#80'][1]}, in a window: {res['#80'][2]})")
print(f"{found}/{tried} files: the quiet word is in main's ranges and outside every #80 range")
