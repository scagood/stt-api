"""Where is the audio main decoded and the PR no longer decodes: the first
range's lead, the last range's tail, or elsewhere? And how far from speech."""
import random, zlib
from collections import Counter
from h import *
from fuzz80 import layout
seed, N, scale = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
for cname in ("v2 ctx5", "whisper"):
    cfg = CONFIGS[cname]
    rng = random.Random(f"wl|{seed}|{cname}|{scale}")
    c = Counter(); lead_kept = []; tail_kept = []
    for i in range(N):
        segs_s, total_s = layout(rng, scale)
        if total_s <= cfg["mx"]:
            continue
        segs = [(int(a * SR), int(b * SR)) for a, b in segs_s]; total = int(total_s * SR)
        wav, words = synth(np.random.default_rng(zlib.crc32(f"{seed}|{cname}|{i}".encode())), segs, total)
        (rm, wm), (rp, wp) = plan(M["main"], wav, segs, cfg), plan(M["pr"], wav, segs, cfg)
        c["layouts"] += 1
        gone = covered(wm, total) & ~covered(wp, total)
        if not gone.any():
            continue
        idx = np.flatnonzero(gone)
        first_speech, last_speech = segs[0][0], segs[-1][1]
        c["lose"] += 1
        c["lose_lead"] += bool((idx < first_speech).any())
        c["lose_tail"] += bool((idx >= last_speech).any())
        c["lose_elsewhere"] += bool(((idx >= first_speech) & (idx < last_speech)).any())
        if (idx < first_speech).any():
            lead_kept.append(((first_speech - wm[0][0]) / SR, (first_speech - wp[0][0]) / SR))
        if (idx >= last_speech).any():
            tail_kept.append(((wm[-1][1] - last_speech) / SR, (wp[-1][1] - last_speech) / SR))
    print(cname, scale, dict(c))
    print("  lead margin main->pr (s):", [(round(a, 2), round(b, 2)) for a, b in lead_kept][:12])
    print("  tail margin main->pr (s):", [(round(a, 2), round(b, 2)) for a, b in tail_kept][:12])
