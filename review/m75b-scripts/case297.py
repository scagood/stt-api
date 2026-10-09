import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/m75b-scripts")
from lib import *
ch_h, _ = MODS["head"]; ch_p, _ = MODS["prev"]
sounds = [(42.0, 0.3, -35.8), (43.69, 1.5, -46.5), (50.85, 1.5, -49.7), (51.25, 0.2, -44.6), (52.81, 0.8, -45.6), (54.92, 1.5, -42.5), (61.44, 0.8, -30.7), (61.54, 0.1, -34.2), (65.01, 0.03, -30.0), (65.21, 0.1, -45.3), (66.69, 0.3, -37.7), (71.94, 0.8, -31.8), (73.63, 0.45, -39.8), (76.86, 0.8, -45.9)]
# reproduce RNG: fuzz used seed=it=297 but sounds' levels were drawn exactly; reuse
wav = build(38.52, sounds, seed=297)
rms = ch_h.frame_rms(wav)
h = ch_h.loud_frames(rms, 0.4, 150); p = ch_p.loud_frames(rms, 0.4, 150)
fr = lambda m: [(round(a*0.02,2), round(b*0.02,2)) for a,b in ch_h.runs(m).tolist()]
print("prev", fr(p)); print("head", fr(h)); print("head&~prev", fr(h & ~p))
