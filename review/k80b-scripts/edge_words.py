"""A quiet word the energy VAD misses, in main's lead or tail margin of an
oversized first/last range. Real _volume_speech_segments, no patching.
usage: edge_words.py N side(lead|tail) word_start word_end [L0 L1 TOT] [cfg]"""
from h import *
N = int(sys.argv[1]); side = sys.argv[2]
qa, qb = int(float(sys.argv[3]) * SR), int(float(sys.argv[4]) * SR)
L0, L1, TOT = (float(x) for x in (sys.argv[5:8] if len(sys.argv) > 7 else (3, 37, 40)))
cfg = CONFIGS[sys.argv[8] if len(sys.argv) > 8 else "v2 ctx5"]
total = int(TOT * SR); segs = [(int(L0 * SR), int(L1 * SR))]
hits = {k: 0 for k in M}; vad_heard = 0; shown = False
for seed in range(N):
    rng = np.random.default_rng(seed)
    wav, words = synth(rng, segs, total)
    wav[qa:qb] += word_signal(rng, qb - qa, db(float(os.environ.get("QDB", "-42")))).astype(np.float32)
    vad = M["main"]._volume_speech_segments(wav)
    vad_heard += any(a < qb and b > qa for a, b in vad)
    res = {}
    for k, m in M.items():
        p = m.plan_chunks(wav, target_sec=cfg["target"], max_sec=cfg["mx"], min_sec=20.0, context_sec=cfg["ctx"])
        res[k] = p
        hits[k] += any(a <= qa and qb <= b for a, b in p.windows)
    if not shown and any(a <= qa and qb <= b for a, b in res["main"].windows) and not any(a <= qa and qb <= b for a, b in res["pr"].windows):
        shown = True
        print("seed", seed, "VAD", sec(vad))
        for k in M: print(f"  {k:5s} ranges {sec(res[k].ranges)} windows {sec(res[k].windows)}")
print(f"{side} word {qa/SR:.2f}-{qb/SR:.2f}s, speech {L0}-{L1}s of {TOT}s, {N} realisations; VAD heard it in {vad_heard}; decoded by: " + ", ".join(f"{k} {v}/{N}" for k, v in hits.items()))
