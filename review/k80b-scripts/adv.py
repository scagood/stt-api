"""Adversarial audio under one long VAD segment (3 s margins): share of forced
cuts inside words, main vs PR, and per-cut gap->word / word->gap moves.
usage: adv.py N variant [cfg]"""
import zlib
from h import *

N = int(sys.argv[1]); variant = sys.argv[2]
cfg = CONFIGS[sys.argv[3] if len(sys.argv) > 3 else "v2 ctx5"]


def lowpass_noise(rng, n, cutoff_hz):
    x = rng.standard_normal(n)
    X = np.fft.rfft(x); f = np.fft.rfftfreq(n, 1 / SR)
    X[f > cutoff_hz] = 0
    y = np.fft.irfft(X, n)
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-12)


def make(rng, L):
    segs = [(3 * SR, int((3 + L) * SR))]
    total = int((6 + L) * SR)
    sp = -20.0
    kw = {}
    if variant == "closures":  # stop closures inside half the words, deeper than shallow word gaps
        kw = dict(closures=dict(p=0.5, dur=(0.06, 0.12), depth_db=-30), gap_db=(-15, -8), gap=(0.03, 0.2))
    elif variant == "connected":  # fast connected speech: short shallow gaps, some closures
        kw = dict(closures=dict(p=0.3, dur=(0.05, 0.1), depth_db=-25), gap_db=(-12, -6), gap=(0.0, 0.12), word=(0.15, 0.6))
    elif variant == "zeros":
        kw = dict(room_db=-400, gap_db=-400)
    elif variant == "quiet":
        sp = -55.0; kw = dict(room_db=-70)
    wav, words = synth(rng, segs, total, speech_db=sp, **kw)
    wav = wav.astype(np.float64)
    n = total
    if variant.startswith("noise"):  # white noise bed X dB under speech
        x = float(variant[5:]); wav += rng.standard_normal(n) * db(sp - x)
    elif variant.startswith("hum"):  # 50 Hz hum + harmonics X dB under speech
        x = float(variant[3:]); t = np.arange(n) / SR
        h = sum(np.sin(2 * np.pi * 50 * k * t) / k for k in (1, 2, 3, 5))
        wav += h / np.sqrt(np.mean(h ** 2)) * db(sp - x)
    elif variant.startswith("rumble"):  # low rumble, X dB relative to speech, slowly swelling
        x = float(variant[6:]); t = np.arange(n) / SR
        r = lowpass_noise(rng, n, 80) * (0.55 + 0.45 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 6)))
        wav += r * db(sp + x)
    elif variant.startswith("music"):  # notes 0.15-0.6 s with short rests, X dB relative to speech
        x = float(variant[5:]); m = np.zeros(n); t0 = 0
        while t0 < n:
            d = int(rng.uniform(0.15, 0.6) * SR); f0 = rng.choice([110, 147, 165, 196, 220, 262, 330])
            tt = np.arange(min(d, n - t0)) / SR
            note = sum(np.sin(2 * np.pi * f0 * k * tt) / k for k in (1, 2, 3)) * np.exp(-tt * rng.uniform(1, 6))
            m[t0:t0 + tt.size] += note
            t0 += d + int(rng.uniform(0.0, 0.15) * SR)
        wav += m / np.sqrt(np.mean(m ** 2)) * db(sp + x)
    elif variant.startswith("clicks"):  # typing: 5 ms clicks every 80-250 ms, X dB relative to speech
        x = float(variant[6:]); t0 = 0
        while t0 < n - 80:
            wav[t0:t0 + 80] += rng.standard_normal(80) * db(sp + x)
            t0 += int(rng.uniform(0.08, 0.25) * SR)
    elif variant.startswith("dc"):
        wav += float(variant[2:])
    elif variant.startswith("clip"):
        g = float(variant[4:]); wav = np.clip(wav * db(g), -1, 1)
    elif variant.startswith("reverb"):  # exponential-decay noise IR, RT60 = X s, half direct half reverberant
        rt = float(variant[6:]); m = int(rt * SR); tt = np.arange(m) / SR
        ir = rng.standard_normal(m) * np.exp(-6.9 * tt / rt); ir *= 1.0 / np.sqrt(np.sum(ir ** 2))
        k = 1 << int(np.ceil(np.log2(n + m)))
        wet = np.fft.irfft(np.fft.rfft(wav, k) * np.fft.rfft(ir, k), k)[:n]
        wav = wav + wet
    elif variant.startswith("breathy"):  # whispered: words are noise only, shallow gaps
        pass
    return wav.astype(np.float32), words, segs, total


tot = {"main": [0, 0], "pr": [0, 0]}; to_word = to_gap = same = 0; diffn = 0; bad = 0
for i in range(N):
    rng = np.random.default_rng(zlib.crc32(f"{variant}|{i}".encode()))
    L = rng.uniform(35, 160)
    wav, words, segs, total = make(rng, L)
    st = np.array([w[0] for w in words]); en = np.array([w[1] for w in words])
    def inw(b):
        j = np.searchsorted(st, b, side="right") - 1
        return bool(j >= 0 and st[j] < b < en[j])
    plans = {k: plan(M[k], wav, segs, cfg) for k in ("main", "pr")}
    bad += bool(check_bounds(*plans["pr"], total, cfg))
    cuts = {}
    for k, (r, w) in plans.items():
        cuts[k] = [b for (a, b), (c, d) in zip(r, r[1:]) if b == c]
        tot[k][0] += len(cuts[k]); tot[k][1] += sum(inw(b) for b in cuts[k])
    if len(cuts["main"]) != len(cuts["pr"]):
        diffn += 1; continue
    for a, b in zip(cuts["main"], cuts["pr"]):
        x, y = inw(a), inw(b)
        to_word += (not x) and y; to_gap += x and not y
print(f"{variant:10s} {N} files: in-word main {tot['main'][1]}/{tot['main'][0]}={tot['main'][1]/max(1,tot['main'][0]):.2f} "
      f"pr {tot['pr'][1]}/{tot['pr'][0]}={tot['pr'][1]/max(1,tot['pr'][0]):.2f}; per cut gap->word {to_word}, word->gap {to_gap}; diff count {diffn}; bounds bad {bad}")
