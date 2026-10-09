"""Replay one fuzz case: case.py SEED CASE MINSIL"""
import sys
import numpy as np
import lib
sys.argv, args = sys.argv[:1] + ["1", sys.argv[1], sys.argv[3]], sys.argv[1:]
import importlib.util
spec = importlib.util.spec_from_file_location("fz", "fuzz.py")
SEED, CASE, MINSIL = int(args[0]), int(args[1]), int(args[2])
src = open("fuzz.py").read().split("rng = np.random.default_rng(SEED)")[0]
ns = {}
exec(compile(src.replace("N, SEED, MINSIL = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])", f"N, SEED, MINSIL = 1, {SEED}, {MINSIL}"), "fz", "exec"), ns)
rng = np.random.default_rng(SEED)
for c in range(CASE + 1):
    floor, P, items = ns["gen"](rng)
sounds = [p for _, _, _, parts in items for p in parts]
wav, off = lib.pause_between(P, sounds, floor=floor, seed=SEED * 100000 + CASE)
print(f"floor {floor:.1f} pause {P:.2f} offset {off:.2f}")
for k, a, b, parts in items:
    print(f"  {k:6s} {a:6.2f}-{b:6.2f} ({b-a:.2f}s)", " ".join(f"[{x:.2f}+{l:.2f}@{d:.0f}]" for x, l, d in parts[:6]), "..." if len(parts) > 6 else "")
for name, mod in (("main", lib.MAIN), ("PR", lib.PR)):
    pl = lib.plan(mod, wav)
    loud = lib.loud_split(mod, wav)
    i0, i1 = int(off / 0.02), int((off + P) / 0.02)
    print(name, "ranges", [(round(a / lib.SR - off, 2), round(b / lib.SR - off, 2)) for a, b in pl.ranges])
    print(name, "loud runs in pause", [(round(a, 2), round(b, 2)) for a, b in lib.frames_runs(loud[i0:i1])])
    print(name, "speech segs (pause-rel)", [(round(a / lib.SR - off, 2), round(b / lib.SR - off, 2)) for a, b in pl.speech if b / lib.SR > off - 1 and a / lib.SR < off + P + 1])
