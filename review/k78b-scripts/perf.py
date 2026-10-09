"""2 h of audio: timing loud_frames / _volume_speech_segments / retime.pauses, main vs PR.
Kinds: 'mixed' (turns, quiet phrases, breaths, click trains), 'clicks' (2 h of a quiet clock ticking
every 0.35 s over room tone with a loud turn every 10 min), 'quiet' (2 h quiet speaker, one loud turn)."""
import sys
import time
import numpy as np
import lib
SR = lib.SR
kind = sys.argv[1]
rng = np.random.default_rng(0)
H = 2 * 3600
if kind == "mixed":
    parts = []
    t = 0
    while t < H:
        L = rng.uniform(5, 40); parts.append(lib.turn(L, -20, rng)); t += L
        P = rng.uniform(2, 15); p = lib.noise(P, -58, rng)
        for _ in range(rng.integers(0, 6)):
            lib.add(p, rng.uniform(0, P - 0.5), rng.uniform(0.05, 0.45), -58 + rng.uniform(7, 20), rng)
        if rng.random() < 0.5:
            lib.add(p, rng.uniform(0, max(0.1, P - 1.5)), rng.uniform(0.5, 1.4), -44, rng)
        parts.append(p); t += P
    wav = np.concatenate(parts)
elif kind == "clicks":
    wav = lib.noise(H, -60, rng)
    n = int(H / 0.35)
    idx = (np.arange(n) * 0.35 * SR).astype(int)
    for i in idx:
        wav[i:i + 960] += (rng.standard_normal(min(960, wav.size - i)) * 10 ** (-45 / 20)).astype(np.float32)
    for k in range(0, H, 600):
        seg = lib.turn(5, -20, rng); wav[k * SR:k * SR + seg.size] += seg
elif kind == "quiet":
    wav = lib.turn(H, -45, rng, syll=0.2, gap=0.5)
    seg = lib.turn(30, -15, rng); wav[:seg.size] += seg
print(kind, "audio", wav.size / SR / 3600, "h")
for tag in ("main", "pr"):
    t0 = time.perf_counter(); s = lib.segs(tag, wav); t1 = time.perf_counter()
    p = lib.pauses(tag, wav); t2 = time.perf_counter()
    print(f"{tag}: splitter {t1 - t0:.2f}s ({len(s)} segs, {sum(b - a for a, b in s) / SR:.0f}s speech)  retime.pauses {t2 - t1:.2f}s ({len(p)} pauses)")
