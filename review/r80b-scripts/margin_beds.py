"""A quiet word in the outer margin of a split first/last range, under a bed.
Real volume VAD (no patching). main vs PR (head): is the word inside a range
(and window) main decodes, and outside every PR range/window?

usage: margin_beds.py SIDE BED BED_DB WORD_DB [N] [bounds]
  SIDE lead|tail; BED none|white|pink|hum|dc|music; BED_DB level of the bed
  (dBFS rms; for dc the offset in dBFS); WORD_DB the quiet word's level.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from synth import MAIN, PR, SR, synth, load

BOUNDS = {"v2": (25.0, 30.0, 5.0), "ctx0": (25.0, 30.0, 0.0), "v3": (60.0, 75.0, 5.0)}


def bed(kind, n, level_db, rng):
    amp = 10 ** (level_db / 20)
    t = np.arange(n) / SR
    if kind == "none":
        return np.zeros(n, np.float32)
    if kind == "white":
        x = rng.standard_normal(n)
    elif kind == "pink":
        f = np.fft.rfftfreq(n, 1 / SR); s = np.fft.rfft(rng.standard_normal(n)); s[1:] /= np.sqrt(f[1:]); s[0] = 0
        x = np.fft.irfft(s, n)
    elif kind == "hum":
        x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 100 * t) + 0.3 * np.sin(2 * np.pi * 150 * t)
    elif kind == "dc":
        return np.full(n, amp, np.float32)
    elif kind == "music":  # chords changing every 0.5 s, with a beat
        x = np.zeros(n)
        for k in range(0, n, SR // 2):
            notes = rng.choice([220, 247, 262, 294, 330, 349, 392, 440], 3)
            seg = slice(k, min(n, k + SR // 2)); tt = t[seg] - t[k]
            for f0 in notes:
                x[seg] += np.sin(2 * np.pi * f0 * tt) * np.exp(-tt * 2)
    x = x / np.sqrt(np.mean(x * x))
    return (x * amp).astype(np.float32)


def main():
    side, kind, bed_db, word_db = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    N = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    target, mx, ctx = BOUNDS[sys.argv[6] if len(sys.argv) > 6 else "v2"]
    lost = lost_win = heard = tried = given = 0
    shown = 0
    for seed in range(N):
        rng = np.random.default_rng(seed)
        speech_len = rng.uniform(36.0, 39.5) if mx == 30.0 else rng.uniform(126.0, 129.5)
        lead = 3.0 if side == "lead" else 0.5
        tail = 3.0 if side == "tail" else 0.3
        total = int((lead + speech_len + tail) * SR)
        seg = (int(lead * SR), int((lead + speech_len) * SR))
        wav, words = synth([seg], total, rng, 20.0, (50, 350))
        if side == "tail":
            at = lead + speech_len + rng.uniform(1.2, 2.2)
        else:
            at = lead - 0.4 - rng.uniform(1.2, 2.2)
        a, b = int(at * SR), int((at + 0.4) * SR)
        w = rng.standard_normal(b - a).astype(np.float32) * 10 ** (word_db / 20)
        r = int(0.03 * SR); w[:r] *= np.linspace(0, 1, r); w[-r:] *= np.linspace(1, 0, r)
        wav[a:b] = w
        wav = (wav + bed(kind, total, bed_db, rng)).astype(np.float32)
        res = {}
        for name, mod in (("main", MAIN), ("pr", PR)):
            p = mod.plan_chunks(wav, target_sec=target, max_sec=mx, min_sec=20.0, context_sec=ctx)
            res[name] = dict(
                heard=any(s < b and e > a for s, e in p.speech),
                inrange=any(r0 <= a and b <= r1 for r0, r1 in p.ranges),
                partial=any(r0 < b and a < r1 for r0, r1 in p.ranges),
                inwin=any(w0 <= a and b <= w1 for w0, w1 in p.windows),
                ranges=[(round(r0 / SR, 2), round(r1 / SR, 2)) for r0, r1 in p.ranges],
                speech=[(round(s / SR, 2), round(e / SR, 2)) for s, e in p.speech],
                span=(p.ranges[0][0], p.ranges[-1][1]),
            )
        tried += 1
        m, q = res["main"], res["pr"]
        heard += m["heard"]
        given += q["span"] != m["span"]
        if m["inrange"] and not q["inrange"]:
            lost += 1
            lost_win += not q["inwin"]
            if shown < 2:
                shown += 1
                print(f"  seed {seed}: word {at:.2f}-{at + 0.4:.2f}s; VAD {m['speech']}\n    main {m['ranges']}\n    pr   {q['ranges']} (partly in a range: {q['partial']}, in a window: {q['inwin']})")
    print(f"{side} bed={kind}@{bed_db} word@{word_db} bounds={sys.argv[6] if len(sys.argv) > 6 else 'v2'}: "
          f"word heard by VAD {heard}/{tried}; outer span changed {given}; word in main's range but not wholly in a PR range: {lost}/{tried} (not in a PR window: {lost_win})")


if __name__ == "__main__":
    main()
