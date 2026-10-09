import numpy as np
import lib
SR = lib.SR
for tail in (0.3, 0.6, 0.9, 1.2, 1.5, 1.8):
    rng = np.random.default_rng(3)
    host = lib.turn(30.0, -20, rng)
    a0, a1 = 0.5, 8.5
    br0 = a1 + 2.9; br1 = br0 + 0.3
    b0 = br1 + 0.45; b1 = b0 + 0.8; P = b1 + tail
    pause = lib.noise(P, -64, rng)
    pause[int(a0 * SR):int(a1 * SR)] += lib.turn(a1 - a0, -36, rng)[: int(a1 * SR) - int(a0 * SR)]
    lib.add(pause, br0, 0.3, -42, rng)
    lib.add(pause, b0, 0.8, -52, rng)
    wav = np.concatenate([host, pause])
    off = host.size / SR
    print(tail, "end", round(wav.size / SR, 2), "B", round(off + b0, 2), round(off + b1, 2))
    for t in ("main", "pr"):
        print("  ", t, lib.loud_runs_sec(t, wav, off + a1 - 0.3, 99), lib.sec(lib.plan(t, wav).ranges))
