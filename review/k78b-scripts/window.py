"""Width of the PR-only failure window in retime.pauses: quiet guest 8 s at -36 dBFS (host -20,
room -64); a straddler starting 2.9 s after the guest stops (one 0.3 s breath, or four 60 ms clicks
0.3 s apart, at -42); 0.45 s later a 0.8 s aside at -48; host resumes TAIL s after the aside.
For each TAIL: is the aside a pause (fully or mostly) on main / PR?"""
import numpy as np
import lib
SR = lib.SR
for name, train in (("breath", [(0.0, 0.3)]), ("4 clicks", [(k * 0.36, 0.06) for k in range(4)])):
    rows = []
    for tail in np.arange(0.8, 3.01, 0.1):
        rng = np.random.default_rng(3)
        host = lib.turn(30, -20, rng)
        a1 = 8.5
        t0 = a1 + 2.9
        te = t0 + train[-1][0] + train[-1][1]
        b0 = te + 0.45; b1 = b0 + 0.8; P = b1 + tail
        pause = lib.noise(P, -64, rng)
        pause[int(0.5 * SR):int(a1 * SR)] += lib.turn(a1 - 0.5, -36, rng)[: int(a1 * SR) - int(0.5 * SR)]
        for at, ln in train:
            lib.add(pause, t0 + at, ln, -42, rng)
        lib.add(pause, b0, 0.8, -48, rng)
        wav = np.concatenate([host, pause, lib.turn(30, -20, rng)])
        off = host.size / SR
        res = []
        for tag in ("main", "pr"):
            ps = lib.pauses(tag, wav)
            inside = sum(max(0, min(b, off + b1) - max(a, off + b0)) for a, b in ps)
            res.append(inside > 0.4)
        rows.append((round(float(tail), 1), res))
    print(name, "aside taken for a pause (main, PR) by tail:", [(t, "".join("X" if r else "." for r in res)) for t, res in rows])
