"""At a setting, is PR's loud_frames always a superset of main's? Count frames loud on main only,
for the splitter (0.4, 150) and retime (0.6, 150). usage: superset.py N SEED MINSIL"""
import sys
import numpy as np
import lib
N, SEED, MINSIL = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
lib.set_minsil(MINSIL)
src = open("fuzz.py").read().split("rng = np.random.default_rng(SEED)")[0]
ns = {}
exec(compile(src.replace("N, SEED, MINSIL = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])", f"N, SEED, MINSIL = 1, {SEED}, {MINSIL}"), "fz", "exec"), ns)
rng = np.random.default_rng(SEED)
bad = {"split": [], "retime": []}
for case in range(N):
    floor, P, items = ns["gen"](rng)
    sounds = [p for _, _, _, parts in items for p in parts]
    wav, off = lib.pause_between(P, sounds, floor=floor, seed=SEED * 100000 + case)
    for kind, fn in (("split", lib.loud_split), ("retime", lib.loud_retime)):
        m, p = fn(lib.MAIN, wav), fn(lib.PR, wav)
        only = m & ~p
        if only.any():
            bad[kind].append((case, int(only.sum()), [(round(a - off, 2), round(b - off, 2)) for a, b in lib.frames_runs(only)][:4]))
print(f"minsil={MINSIL} seed={SEED} N={N}: main-only loud frames: split {len(bad['split'])} cases, retime {len(bad['retime'])} cases")
for k in bad:
    for b in bad[k][:6]:
        print(" ", k, b)
