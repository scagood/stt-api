"""Repro (args: TAIL BREATH_DB ASIDE_DB [end]): quiet guest 8 s at -36 dBFS under a -20 dBFS host, room tone -64.
after the guest stops, one 0.3 s breath (-50) reaching past the turn's 3 s reach; 0.45 s after it
a 0.8 s quieter aside (-48 dBFS, 12 dB under the guest); the host resumes 1.2 s later.
Compare loud frames, retime.pauses, the plan, and what retime() does to the aside's word."""
import sys
import numpy as np
import lib_b as lib
SR = lib.SR
TAIL = float(sys.argv[1]) if len(sys.argv) > 1 else 1.2
AT_END = "end" in sys.argv
BR = float(sys.argv[2]) if len(sys.argv) > 2 else -42
BL = float(sys.argv[3]) if len(sys.argv) > 3 else -48
rng = np.random.default_rng(3)
host = lib.turn(30, -20, rng)
a0, a1 = 0.5, 8.5
br0 = a1 + 2.9; br1 = br0 + 0.3
b0 = br1 + 0.45; b1 = b0 + 0.8; P = b1 + TAIL
pause = lib.noise(P, -64, rng)
pause[int(a0 * SR):int(a1 * SR)] += lib.turn(a1 - a0, -36, rng)[: int(a1 * SR) - int(a0 * SR)]
lib.add(pause, br0, 0.3, BR, rng)
lib.add(pause, b0, 0.8, BL, rng)
wav = np.concatenate([host, pause] + ([] if AT_END else [lib.turn(30, -20, rng)]))
off = host.size / SR
print(f"guest {off + a0:.2f}-{off + a1:.2f}, reach to {off + a1 + 3:.2f}; breath {off + br0:.2f}-{off + br1:.2f}; aside {off + b0:.2f}-{off + b1:.2f}; host resumes {off + P:.2f}")
for tag in ("main", "old", "head"):
    print(tag, "retime loud ", lib.loud_runs_sec(tag, wav, off + a1 - 0.5, off + P + 0.5, ratio=0.6))
    ps = [(round(a, 2), round(b, 2)) for a, b in lib.pauses(tag, wav) if b > off + a1 and a < off + P]
    print(tag, "retime pauses", ps)
    print(tag, "split  loud ", lib.loud_runs_sec(tag, wav, off + a1 - 0.5, off + P + 0.5, ratio=0.4))
    pl = lib.plan(tag, wav)
    print(tag, "ranges", lib.sec(pl.ranges), "aside decoded", lib.covered(pl.ranges, int((off + b0) * SR), int((off + b1) * SR)) / SR)
    # Retime one chunk's words: guest's last word, then the aside as one word "Right.", then host's first word.
    words = [
        {"word": "okay", "start": off + a1 - 0.4, "end": off + a1 - 0.05},
        {"word": "Right.", "start": off + b0 + 0.05, "end": off + b1 - 0.05},
        {"word": "So", "start": off + P + 0.05, "end": off + P + 0.3},
    ]
    gaps = lib.pauses(tag, wav)
    ends = [e for _s, e in gaps]
    low, high = off - 5, off + P + 10
    out = lib.MODS[tag][1].retime(words, lib.MODS[tag][1].within(gaps, ends, low, high), low, high)
    print(tag, "retimed", [(w["word"], round(w["start"], 2), round(w["end"], 2)) for w in out])



