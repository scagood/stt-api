"""Two independent speech-like generators.
A: the earlier review's (k80a synth.py): noise words 0.15-0.5 s, +-4 dB, syllable dips 6-10 dB, gaps 50-350 ms GAP dB down, room -65 dBFS.
B: new: harmonic voiced syllables (f0 90-230 Hz) with Hann envelopes, 1-3 per word, fricative noise bursts,
   lognormal gaps (median 110 ms), gaps 15-30 dB down, pink room tone at -62 dBFS."""
import numpy as np
from common import SR


def gen_a(segs, total, nrng, gap_db=20.0, gap_ms=(50, 350), base_db=-20.0, room_db=-65.0, word_sd=4.0):
    env = np.full(total, 10 ** (room_db / 20), dtype=np.float32)
    words = []
    gap_amp = max(10 ** ((base_db - gap_db) / 20), 10 ** (room_db / 20))
    for s, e in segs:
        t = s
        while t < e:
            wl = int(nrng.uniform(0.15, 0.5) * SR); we = min(e, t + wl); n = we - t
            lvl = 10 ** ((base_db + nrng.normal(0, word_sd)) / 20)
            shape = np.ones(n, dtype=np.float32)
            nsyl = nrng.integers(1, 4)
            for k in range(1, nsyl):
                c = n * k // nsyl; w = int(nrng.uniform(0.03, 0.06) * SR)
                shape[max(0, c - w // 2): c + w // 2] = 10 ** (-nrng.uniform(6, 10) / 20)
            r = min(int(0.01 * SR), n // 2)
            if r:
                shape[:r] *= np.linspace(0.1, 1, r, dtype=np.float32); shape[n - r:] *= np.linspace(1, 0.1, r, dtype=np.float32)
            env[t:we] = lvl * shape; words.append((t, we))
            ge = min(e, we + int(nrng.uniform(*gap_ms) / 1000 * SR)); env[we:ge] = gap_amp; t = ge
    return nrng.standard_normal(total, dtype=np.float32) * env, words


_PINK = {}


def pink(n, nrng):
    """a random slice of a cached 2^23-sample pink buffer (reversed half the time)"""
    if "buf" not in _PINK:
        _PINK["buf"] = _pink(1 << 23, np.random.default_rng(12345))
    buf = _PINK["buf"]
    if n > buf.size:
        return _pink(n, nrng)
    a = int(nrng.integers(0, buf.size - n + 1))
    out = buf[a:a + n].copy()
    return out[::-1].copy() if nrng.random() < 0.5 else out


def _pink(n, nrng):
    m = 1 << max(4, int(np.ceil(np.log2(max(n, 2)))))
    x = nrng.standard_normal(m)
    X = np.fft.rfft(x); f = np.arange(X.size, dtype=float); f[0] = 1
    y = np.fft.irfft(X / np.sqrt(f), m)[:n]
    return (y / (np.std(y) + 1e-12)).astype(np.float32)


def word_b(n, nrng):
    """one word of n samples, unit-ish rms: 1-3 voiced syllables, maybe a fricative."""
    out = np.zeros(n, dtype=np.float32)
    nsyl = int(nrng.integers(1, 4)) if n > 0.2 * SR else 1
    bounds = np.sort(nrng.uniform(0, 1, nsyl - 1)) if nsyl > 1 else np.array([])
    edges = np.concatenate(([0], (bounds * n).astype(int), [n]))
    t = np.arange(n) / SR
    f0 = nrng.uniform(90, 230)
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 16:
            continue
        m = b - a
        tt = t[a:b]
        f = f0 * (1 + 0.1 * np.sin(2 * np.pi * nrng.uniform(1, 4) * tt))
        ph = 2 * np.pi * np.cumsum(f) / SR
        sig = sum((0.8 ** k) * np.sin(k * ph + nrng.uniform(0, 6.28)) for k in range(1, 12))
        envl = np.hanning(m) ** nrng.uniform(0.5, 1.2) * 10 ** (nrng.normal(0, 2) / 20)
        out[a:b] += (sig * envl).astype(np.float32)
    if nrng.random() < 0.4:  # fricative
        m = int(min(n, nrng.uniform(0.04, 0.12) * SR)); a = int(nrng.integers(0, n - m + 1))
        out[a:a + m] += nrng.standard_normal(m).astype(np.float32) * 0.5 * np.hanning(m).astype(np.float32)
    return out / (np.sqrt(np.mean(out ** 2)) + 1e-9)


def gen_b(segs, total, nrng, base_db=-20.0, room_db=-62.0, word_sd=3.0, gap_db=(15, 30)):
    wav = pink(total, nrng) * 10 ** (room_db / 20)
    words = []
    for s, e in segs:
        t = s
        while t < e:
            wl = int(nrng.uniform(0.12, 0.55) * SR); we = min(e, t + wl)
            if we - t > 32:
                wav[t:we] += word_b(we - t, nrng) * 10 ** ((base_db + nrng.normal(0, word_sd)) / 20)
                words.append((t, we))
            g = int(np.clip(nrng.lognormal(np.log(0.11), 0.6), 0.03, 0.39) * SR)
            ge = min(e, we + g)
            if ge > we:
                wav[we:ge] += nrng.standard_normal(ge - we).astype(np.float32) * 10 ** ((base_db - nrng.uniform(*gap_db)) / 20)
            t = ge
    return wav, words


def quiet_word(wav, a, b, level_db, nrng, kind):
    if kind == "a":
        wav[a:b] = nrng.standard_normal(b - a).astype(np.float32) * 10 ** (level_db / 20)
    else:
        wav[a:b] += word_b(b - a, nrng) * 10 ** (level_db / 20)
