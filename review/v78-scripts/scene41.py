import sys
MS = int(sys.argv[1]); IDX = int(sys.argv[2]); SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 5
sys.argv = [sys.argv[0], "0", str(SEED), str(MS)] + sys.argv[4:]
import numpy as np
import scenefuzz as F
lib = F.lib; SR = lib.SR
rng = np.random.default_rng(SEED)
for i in range(IDX + 1):
    wav, speech, non = F.scene(rng)
sp = [(int(a * SR), int(b * SR)) for a, b in speech]
for tag in ("main", "head"):
    pl = lib.plan(tag, wav)
    print(tag, "ranges", lib.sec(pl.ranges))
    print(tag, "windows", lib.sec(pl.windows))
    for (ws, we), (rs, re) in zip(pl.windows, pl.ranges):
        for e in ((ws,) if ws != rs else ()) + ((we,) if we != re else ()):
            hit = [(round(a / SR, 2), round(b / SR, 2)) for a, b in sp if a < e < b]
            if hit: print("   window edge in speech", round(e / SR, 2), hit)
    m = lib.measure if hasattr(lib, "measure") else None
r = {t: F.measure(t, wav, speech, non) for t in ("main", "head")}
print({t: (r[t]["got"] / SR, r[t]["lost"]) for t in r})
for tag in ("main", "head"):
    ls = lib.loud(tag, wav)
    print(tag, "speech segs", len(lib.MODS[tag][0]._volume_speech_segments(wav)))
dm = lib.loud("main", wav); dh = lib.loud("head", wav)
extra = lib.MODS["main"][0].runs(dh & ~dm)
print("head-only loud frames (s):", [(round(a * 0.02, 2), round(b * 0.02, 2)) for a, b in extra.tolist()])
print("speech near those:", [s for s in speech if any(a * 0.02 - 4 < s[0] < b * 0.02 + 4 for a, b in extra.tolist())])
print("nonspeech near those:", [(round(a,2), round(b,2)) for a, b in non if any(a2 * 0.02 - 4 < a < b2 * 0.02 + 4 for a2, b2 in extra.tolist())])
