import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/m75b-scripts")
from lib import *
ch_h, _ = MODS["head"]
fr = lambda m: [(round(a*0.02,2), round(b*0.02,2)) for a,b in ch_h.runs(m).tolist()]
# 20 s pause; quiet phrase 1.0 s at -40 at 45.0-46.0; quiet word 0.4 s at -40 starting 2.9 s after phrase end (48.9-49.3)
for gap in (2.7, 2.8, 2.9, 2.95):
    w0 = 46.0 + gap
    sounds = [(45.0, 1.0, -40), (w0, 0.4, -40)]
    wav = build(20, sounds, seed=3)
    rms = ch_h.frame_rms(wav)
    print(f"gap {gap}: word {w0:.2f}-{w0+0.4:.2f}")
    for n in ("prev", "head"):
        m = loudmask(n, rms, 0.4, 150)
        print(f"   {n} loud@0.4:", [r for r in fr(m) if 40 < r[0] < 60])
        print(f"   {n} plan:", plan(n, wav))
        m6 = loudmask(n, rms, 0.6, 150)
        print(f"   {n} pauses:", [p for p in pauses(n, wav) if p[1] > 40 and p[0] < 60])
