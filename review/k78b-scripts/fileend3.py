"""File ends in a quiet stretch: host turn (length H), quiet guest 8 s at -36, a straddling 0.3 s
breath (-42) at the guest's 3 s reach, then a quieter 0.8 s phrase B (-52) and the file ends.
Does the plan decode B? Sweep H and the tail so main re-listens and PR doesn't."""
import numpy as np
import lib
SR = lib.SR
res = []
for H in np.arange(15, 95, 0.25):
    for tail in (1.55, 1.6, 1.65, 1.7):
        rng = np.random.default_rng(3)
        host = lib.turn(float(H), -20, rng)
        a0, a1 = 0.5, 8.5
        br0 = a1 + 2.9; br1 = br0 + 0.3
        b0 = br1 + 0.45; b1 = b0 + 0.8; P = b1 + tail
        pause = lib.noise(P, -64, rng)
        pause[int(a0 * SR):int(a1 * SR)] += lib.turn(a1 - a0, -36, rng)[: int(a1 * SR) - int(a0 * SR)]
        lib.add(pause, br0, 0.3, -42, rng)
        lib.add(pause, b0, 0.8, -52, rng)
        wav = np.concatenate([host, pause])
        off = host.size / SR
        B = (int((off + b0) * SR), int((off + b1) * SR))
        cov = {t: lib.covered(lib.plan(t, wav).ranges, *B) / SR for t in ("main", "pr")}
        if abs(cov["main"] - cov["pr"]) > 0.01:
            res.append((float(H), tail, round(cov["main"], 2), round(cov["pr"], 2)))
print("cases where B's decoded seconds differ (H, tail, main, pr):", len(res))
for r in res[:20]:
    print(r)
