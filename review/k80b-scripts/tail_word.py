"""Quiet last word 1.3-1.7 s after the speech, which the energy VAD misses.
Real _speech_segments (PARAKEET_VAD=volume default), no patching."""
from h import *
cfg = CONFIGS["v2 ctx5"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
total = int(40 * SR)
segs = [(int(3 * SR), int(37 * SR))]
qa, qb = int(38.3 * SR), int(38.7 * SR)
for k, m in M.items():
    m._speech_segments = m.__dict__["_speech_segments"]
stats = {"main": 0, "pr": 0}
first = None
n_ok = 0
for seed in range(N):
    rng = np.random.default_rng(seed)
    wav, words = synth(rng, segs, total)
    wav[qa:qb] += word_signal(rng, qb - qa, db(-42)).astype(np.float32)  # quiet "bye", 22 dB under the speech
    res = {}
    for k, m in M.items():
        vad = m._volume_speech_segments(wav)
        p = m.plan_chunks(wav, target_sec=cfg["target"], max_sec=cfg["mx"], min_sec=20.0, context_sec=cfg["ctx"])
        hit = any(a <= qa and qb <= b for a, b in p.windows)
        res[k] = (p.ranges, p.windows, vad, hit)
        stats[k] += hit
    if res["main"][3] and not res["pr"][3] and first is None:
        first = seed
        print("seed", seed, "VAD", sec(res["main"][2]))
        print("  main ranges", sec(res["main"][0]), "windows", sec(res["main"][1]))
        print("  pr   ranges", sec(res["pr"][0]), "windows", sec(res["pr"][1]))
print(f"N={N}: quiet last word (38.3-38.7 s) inside a decoded window: main {stats['main']}/{N}, pr {stats['pr']}/{N}")
