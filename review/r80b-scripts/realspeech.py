"""Main vs #80 on real LibriVox speech, cuts classified by wav2vec2 word times.
usage: realspeech.py CH [CH ...]   (reads r80b-dl/holmes_CH.f32 and words_CH.npy)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from synth import MAIN, PR, CONFIGS, SR, forced_cuts, cover

DL = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/r80b-dl"


def classify(x, words, tol):
    t = x / SR
    i = np.searchsorted(words[:, 0], t, side="right") - 1
    return bool(i >= 0 and words[i, 0] + tol < t < words[i, 1] - tol)


def plan(mod, wav, cfg, gate, segs=None):
    mod.VAD_GATE_DB = gate
    if segs is not None:
        mod._speech_segments = lambda _w: segs
    else:
        mod._speech_segments = mod._speech_segments_orig
    p = mod.plan_chunks(wav, target_sec=cfg["target"], max_sec=cfg["mx"], min_sec=20.0, context_sec=cfg["ctx"])
    return p


for mod in (MAIN, PR):
    mod._speech_segments_orig = mod._speech_segments

chapters = sys.argv[1:]
scen = [("clean, own gate", None, None), ("clean, gate -60 dBFS (one segment)", None, -60.0),
        ("noise SNR 25 dB, gate under noise", 25.0, "under"), ("noise SNR 15 dB, gate under noise", 15.0, "under"),
        ("noise SNR 15 dB, own gate", 15.0, None), ("noise SNR 8 dB, own gate", 8.0, None),
        ("noise SNR 8 dB, gate under noise", 8.0, "under"), ("noise SNR 12 dB, gate under noise", 12.0, "under"),
        ("noise SNR 20 dB, gate under noise", 20.0, "under"),
        ("pauses cut to 0.25 s, own gate", "squeeze", None), ("pauses cut to 0.25 s, SNR 15, own gate", "squeeze15", None),
        ("pauses cut to 0.25 s, SNR 10, own gate", "squeeze10", None)]
TOT = {}
for ch in chapters:
    clean = np.fromfile(f"{DL}/holmes_{ch}.f32", dtype=np.float32)
    words = np.load(f"{DL}/words_{ch}.npy")
    rms = MAIN.frame_rms(clean)
    speech_level = float(np.sqrt(np.mean(rms[rms > np.percentile(rms, 50)] ** 2)))
    print(f"ch {ch}: {clean.size/SR:.0f}s, words {len(words)}, word share {(words[:,1]-words[:,0]).sum()/(clean.size/SR):.0%}, speech level {20*np.log10(speech_level):.1f} dBFS", flush=True)
    rng = np.random.default_rng(int("".join(c for c in ch if c.isdigit())))
    # pauses (wav2vec2 word gaps) longer than 0.25 s shortened to 0.25 s: speech with no pause VAD can cut at
    keep = np.ones(clean.size, dtype=bool)
    for (s0, e0), (s1, e1) in zip(words[:-1], words[1:]):
        if s1 - e0 > 0.25:
            a, b = int((e0 + 0.125) * SR), int((s1 - 0.125) * SR)
            keep[a:b] = False
    kept_before = np.concatenate(([0], np.cumsum(keep)))
    sq_words = np.stack([kept_before[(words[:, 0] * SR).astype(int)], kept_before[(words[:, 1] * SR).astype(int)]], axis=1) / SR
    squeezed = clean[keep]
    all_words = words
    for sname, snr, gate in scen:
        words = all_words
        if snr is None:
            wav = clean
        elif isinstance(snr, str):
            wav, words = squeezed, sq_words
            if snr != "squeeze":
                wav = squeezed + rng.standard_normal(squeezed.size).astype(np.float32) * speech_level * 10 ** (-float(snr[7:]) / 20)
        else:
            noise_amp = speech_level * 10 ** (-snr / 20)
            wav = clean + (rng.standard_normal(clean.size).astype(np.float32) * noise_amp)
            if gate == "under":
                gate = 20 * np.log10(noise_amp) - 6
        for cname in ("v2 ctx5", "v3 ctx5", "whisper"):
            cfg = CONFIGS[cname]
            pm, pp = plan(MAIN, wav, cfg, gate), plan(PR, wav, cfg, gate)
            segs = pm.speech
            T = TOT.setdefault((sname, cname), Counter())
            fm, fp = forced_cuts(pm.ranges, segs), forced_cuts(pp.ranges, segs)
            T["main_n"] += len(pm.ranges); T["pr_n"] += len(pp.ranges)
            T["main_forced"] += len(fm); T["pr_forced"] += len(fp)
            for tol, tag in ((0.0, ""), (0.02, "20")):
                T["main_inword" + tag] += sum(classify(x, words, tol) for x in fm)
                T["pr_inword" + tag] += sum(classify(x, words, tol) for x in fp)
            if len(fm) == len(fp):
                for x, y in zip(fm, fp):
                    a, b = classify(x, words, 0.0), classify(y, words, 0.0)
                    T[f"{'W' if a else 'g'}->{'W' if b else 'g'}"] += 1
                    T["moved_sum"] += abs(y - x) / SR; T["moved_n"] += 1; T["unmoved"] += x == y
            else:
                T["fc_differs"] += 1
            own = int((cfg["mx"] - 2 * cfg["ctx"]) * SR)
            for p, tag in ((pm, "main"), (pp, "pr")):
                T[tag + "_bad"] += sum(b - a > own for a, b in p.ranges) + sum(b - a > int(cfg["mx"] * SR) for a, b in p.windows) \
                    + sum(not (wa <= a and b <= wb) for (a, b), (wa, wb) in zip(p.ranges, p.windows))
                T[tag + "_short"] += sum(b - a < 2 * SR for a, b in p.ranges)
                T[tag + "_lost_s"] += sum((e - s) - sum(max(0, min(b, e) - max(a, s)) for a, b in p.ranges) for s, e in segs) / SR
            cm, cp = cover(pm.ranges, wav.size), cover(pp.ranges, wav.size)
            d = cm & ~cp
            T["drop_s"] += d.sum() / SR
            # words (wav2vec2) in main's ranges but not in #80's
            for s, e in words:
                a, b = int(s * SR), int(e * SR)
                if d[a:b].any():
                    T["words_dropped"] += 1
    if os.environ.get("PERCH"):
        for (sname, cname), T in TOT.items():
            print(" ", ch, sname, cname, dict(T), flush=True)
for (sname, cname), T in TOT.items():
    f = T.__getitem__
    print(f"{sname:36s} {cname}: pieces {f('main_n')}->{f('pr_n')} forced {f('main_forced')}->{f('pr_forced')} "
          f"in word {f('main_inword')} ({f('main_inword')/max(1,f('main_forced')):.0%}) -> {f('pr_inword')} ({f('pr_inword')/max(1,f('pr_forced')):.0%}); "
          f">=20ms inside {f('main_inword20')/max(1,f('main_forced')):.0%} -> {f('pr_inword20')/max(1,f('pr_forced')):.0%} | "
          f"W->g {f('W->g')} g->W {f('g->W')} W->W {f('W->W')} g->g {f('g->g')} moved {f('moved_sum')/max(1,f('moved_n')):.2f}s unmoved {f('unmoved')} | "
          f"<2s {f('main_short')}->{f('pr_short')} lost {f('main_lost_s'):.1f}->{f('pr_lost_s'):.1f}s bad {f('main_bad')}->{f('pr_bad')} "
          f"dropped {f('drop_s'):.2f}s words dropped {f('words_dropped')} fc_differs {f('fc_differs')}", flush=True)
