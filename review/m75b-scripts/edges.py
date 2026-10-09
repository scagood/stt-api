import sys, warnings; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/m75b-scripts")
from lib import *
warnings.simplefilter("error")
rng = np.random.default_rng(5)
cases = {
 "zeros 120s": np.zeros(120 * SR, np.float32),
 "noise -55 120s": (rng.standard_normal(120 * SR) * db(-55)).astype(np.float32),
 "one sample": np.ones(1, np.float32) * 0.1,
 "319 samples": (rng.standard_normal(319) * 0.1).astype(np.float32),
 "320 samples": (rng.standard_normal(320) * 0.1).astype(np.float32),
 "5 frames": (rng.standard_normal(1600) * 0.1).astype(np.float32),
 "quiet sound at very start (0-0.6s -40) then 120s room tone then turn": build(120, [(0.0, 0.6, -40), (0.7, 0.3, -42), (2.9, 0.3, -42)], first=False),
 "quiet sound at very end": build(120, [(159.4, 0.6, -40), (157.0, 0.3, -42)], last=False),
 "float64 input": build(20, [(45, 0.6, -40)]).astype(np.float64),
}
for name, wav in cases.items():
    print("==", name)
    for n in ("main", "prev", "head"):
        ch, rt = MODS[n]
        try:
            r = plan(n, wav); p = pauses(n, wav)
            rms = ch.frame_rms(wav)
            m = loudmask(n, rms, 0.4, 150) if rms.size else None
            print(f"  {n}: plan {r[:4]} pauses {[(float(a), float(b)) for a, b in p][:6]} loud dtype {None if m is None else m.dtype}")
        except Exception as e:
            print(f"  {n}: EXC {type(e).__name__}: {e}")
# direct loud_frames edge calls
ch = MODS["head"][0]
for rms in (np.array([], np.float32), np.full(10, 1e-6, np.float32), np.full(300, 1e-6, np.float32), np.r_[np.full(30, 0.01), np.full(300, 1e-4)].astype(np.float32)):
    for rel in (1, 5, 25, 150, 10**9):
        try:
            m = ch.loud_frames(rms, 0.4, rel); print("head loud_frames n", rms.size, "rel", rel, "->", int(m.sum()), m.dtype)
        except Exception as e:
            print("head loud_frames n", rms.size, "rel", rel, "EXC", type(e).__name__, e)
