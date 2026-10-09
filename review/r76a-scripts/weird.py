import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/r76a-scripts")
from collections import Counter
from multiprocessing import Pool
import fuzz76
from fuzz76 import work, init
W = {
  "v2 ctx7.5": dict(target=25.0, mx=30.0, ctx=7.5, mn=20.0),
  "ctx10 own10": dict(target=25.0, mx=30.0, ctx=10.0, mn=20.0),
  "t5 mx30 ctx5": dict(target=5.0, mx=30.0, ctx=5.0, mn=20.0),
  "t12 mx15 ctx0 min12": dict(target=12.0, mx=15.0, ctx=0.0, mn=12.0),
  "v3 ctx18.75": dict(target=60.0, mx=75.0, ctx=18.75, mn=20.0),
}
if __name__ == '__main__':
    jobs = [(c, cfg, g, s, "R76A-weird", lo, lo + 2000) for c, cfg in W.items() for g in ("a", "b") for s in (1.0, 1.6) for lo in range(0, 6000, 2000)]
    agg = {}
    with Pool(4, initializer=init) as p:
        for cname, gen, scale, tot, cmp, ex in p.imap_unordered(work, jobs):
            A = agg.setdefault(cname, [Counter(), Counter(), Counter(), {}])
            A[0].update(tot["main"]); A[1].update(tot["head"]); A[2].update(cmp)
            for k, v in ex.items(): A[3].setdefault(k, v)
    for k, (m, h, c, ex) in agg.items():
        print("==", k); print("   main", dict(m)); print("   head", dict(h)); print("   head vs main", dict(sorted(c.items())))
        for kk, v in ex.items(): print("   ex", kk, str(v)[:600])
