"""Main vs head on LibriVox Holmes ch.3 (raw, has +0.094 DC), ch.3 DC-removed, ch.5; wav2vec2 word times.
Forced cuts in words, files worse, coverage dropped, context-window edges in words. usage: realcheck.py CH[,CH..]"""
import os, sys, bisect
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from common import MAIN, HEAD, SR, plan, CONFIGS
from gens import pink

DL = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/r80a-dl"
WDL = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/k80a-dl"


def inw(x, W, starts, tol=0.0):
    t = x / SR
    i = bisect.bisect_right(starts, t) - 1
    return i >= 0 and W[i][0] + tol < t < W[i][1] - tol


SCEN = ["clean own gate", "clean gate -60", "white SNR25 gate<noise", "white SNR20 gate<noise", "white SNR15 gate<noise", "white SNR12 gate<noise",
        "white SNR15 own gate", "white SNR8 own gate", "pink -40 gate -45", "pink -35 gate -45",
        "squeeze own gate", "squeeze SNR15 own gate", "squeeze SNR10 own gate"]
TOT = {}
files_worse = Counter(); files = Counter(); files_better = Counter()
for ch in sys.argv[1].split(","):
    raw = np.fromfile(f"{DL}/holmes_{ch[:2]}.f32", dtype=np.float32)
    clean = raw - raw.mean() if ch.endswith("dc") else raw
    words = np.load(f"{WDL}/words_{ch}.npy")
    rms = MAIN.frame_rms(clean - clean.mean())
    lvl = float(np.sqrt(np.mean(rms[rms > np.percentile(rms, 50)] ** 2)))
    keep = np.ones(clean.size, dtype=bool)
    for (s0, e0), (s1, e1) in zip(words[:-1], words[1:]):
        if s1 - e0 > 0.25:
            keep[int((e0 + 0.125) * SR): int((s1 - 0.125) * SR)] = False
    kb = np.concatenate(([0], np.cumsum(keep)))
    sqw = np.stack([kb[(words[:, 0] * SR).astype(int)], kb[(words[:, 1] * SR).astype(int)]], axis=1) / SR
    print(f"ch {ch}: {clean.size/SR:.0f}s words {len(words)} speech level {20*np.log10(lvl):.1f} dBFS mean {clean.mean():+.4f}", flush=True)
    rng = np.random.default_rng(int(ch[:2]) * 31 + len(ch))
    for sname in SCEN:
        W = words; gate = None
        if sname.startswith("squeeze"):
            base = clean[keep]; W = sqw
        else:
            base = clean
        wav = base
        if "SNR" in sname:
            snr = float(sname.split("SNR")[1].split()[0])
            amp = lvl * 10 ** (-snr / 20)
            wav = base + rng.standard_normal(base.size).astype(np.float32) * amp
            if "gate<noise" in sname:
                gate = 20 * np.log10(amp) - 6
        elif sname.startswith("pink"):
            nl = float(sname.split()[1])
            wav = base + pink(base.size, rng) * 10 ** (nl / 20); gate = -45.0
        elif sname == "clean gate -60":
            gate = -60.0
        starts = W[:, 0].tolist()
        for cfg in CONFIGS:
            ps = {}
            for nm, mod in (("m", MAIN), ("h", HEAD)):
                ps[nm] = plan(mod, wav, cfg, gate=gate)
            T = TOT.setdefault((sname, cfg), Counter())
            segs = ps["m"].speech
            sst = [s for s, _ in segs]
            def forced(p):
                return [b for (a, b), (c, d) in zip(p.ranges, p.ranges[1:]) if b == c and any(s < b < e for s, e in p.speech)]
            fm, fh = forced(ps["m"]), forced(ps["h"])
            T["m_n"] += len(ps["m"].ranges); T["h_n"] += len(ps["h"].ranges)
            T["m_f"] += len(fm); T["h_f"] += len(fh)
            im = sum(inw(x, W, starts) for x in fm); ih = sum(inw(x, W, starts) for x in fh)
            T["m_w"] += im; T["h_w"] += ih
            T["m_deep"] += sum(inw(x, W, starts, 0.04) for x in fm); T["h_deep"] += sum(inw(x, W, starts, 0.04) for x in fh)
            key = (sname, cfg)
            files[key] += 1; files_worse[key] += ih > im; files_better[key] += ih < im
            if len(fm) == len(fh):
                for x, y in zip(fm, fh):
                    a, b = inw(x, W, starts), inw(y, W, starts)
                    T[f"{'W' if a else 'g'}->{'W' if b else 'g'}"] += 1
            own = int((CONFIGS[cfg][1] - 2 * CONFIGS[cfg][2]) * SR)
            for nm in ("m", "h"):
                p = ps[nm]
                T[nm + "_bad"] += sum(b - a > own for a, b in p.ranges) + sum(b - a > int(CONFIGS[cfg][1] * SR) for a, b in p.windows) + sum(not (wa <= a and b <= wb) for (a, b), (wa, wb) in zip(p.ranges, p.windows))
                T[nm + "_short"] += sum(b - a < 2 * SR for a, b in p.ranges)
                T[nm + "_lost"] += sum((e - s) - sum(max(0, min(b, e) - max(a, s)) for a, b in p.ranges) for s, e in p.speech)
                # context edges: window edges beyond the range
                E = [wa for (a, b), (wa, wb) in zip(p.ranges, p.windows) if wa < a] + [wb for (a, b), (wa, wb) in zip(p.ranges, p.windows) if wb > b]
                T[nm + "_E"] += len(E)
                T[nm + "_Ew"] += sum(inw(x, W, starts) for x in E)
                T[nm + "_Es"] += sum(any(s < x < e for s, e in p.speech) for x in E)
            cm = np.zeros(wav.size, bool); chh = np.zeros(wav.size, bool)
            for a, b in ps["m"].ranges: cm[a:b] = True
            for a, b in ps["h"].ranges: chh[a:b] = True
            d = cm & ~chh
            T["drop_s"] += d.sum() / SR
            T["drop_words"] += sum(bool(d[int(s * SR):int(e * SR)].any()) for s, e in W) if d.any() else 0
for (sname, cfg), T in TOT.items():
    f = T.__getitem__; k = (sname, cfg)
    print(f"{sname:24s} {cfg}: pieces {f('m_n')}->{f('h_n')} forced {f('m_f')}->{f('h_f')} in word {f('m_w')} ({f('m_w')/max(1,f('m_f')):.0%}) -> {f('h_w')} ({f('h_w')/max(1,f('h_f')):.0%}) "
          f">40ms {f('m_deep')}->{f('h_deep')} | W->g {f('W->g')} g->W {f('g->W')} | files worse/better {files_worse[k]}/{files_better[k]} of {files[k]} | "
          f"<2s {f('m_short')}->{f('h_short')} lost {f('m_lost')}->{f('h_lost')} bad {f('m_bad')}->{f('h_bad')} dropped {f('drop_s'):.2f}s words {f('drop_words')} | "
          f"ctx edges {f('m_E')}->{f('h_E')} in speech {f('m_Es')}->{f('h_Es')} in word {f('m_Ew')} ({f('m_Ew')/max(1,f('m_E')):.0%}) -> {f('h_Ew')} ({f('h_Ew')/max(1,f('h_E')):.0%})", flush=True)
