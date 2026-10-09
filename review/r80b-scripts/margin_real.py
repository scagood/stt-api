"""Real LibriVox speech (Holmes ch. 5, pauses squeezed to 0.25 s so the
volume VAD hears one long stretch), then a real word from the same chapter,
attenuated, 1.2-2.2 s past (tail) or before (lead) the speech, all over a
bed. Real volume VAD. Is the word in main's ranges but out of head's?

usage: margin_real.py SIDE BED BED_DB WORD_REL_DB [N] [bounds]
  WORD_REL_DB: the word's RMS relative to the bed's (dB); with BED none, the word's dBFS.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from synth import MAIN, PR, SR
from margin_beds import bed, BOUNDS

DL = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/r80b-dl"
clean = np.fromfile(f"{DL}/holmes_05.f32", dtype=np.float32)
words = np.load(f"{DL}/words_05.npy")
keep = np.ones(clean.size, dtype=bool)
for (s0, e0), (s1, e1) in zip(words[:-1], words[1:]):
    if s1 - e0 > 0.25:
        keep[int((e0 + 0.125) * SR): int((s1 - 0.125) * SR)] = False
kb = np.concatenate(([0], np.cumsum(keep)))
sq = clean[keep]
sqw = np.stack([kb[(words[:, 0] * SR).astype(int)], kb[(words[:, 1] * SR).astype(int)]], axis=1)
long_words = [(int(a * SR), int(b * SR)) for a, b in words if 0.35 <= b - a <= 0.6]


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12)


def main():
    side, kind, bed_db, rel = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    N = int(sys.argv[5]) if len(sys.argv) > 5 else 60
    bname = sys.argv[6] if len(sys.argv) > 6 else "v2"
    target, mx, ctx = BOUNDS[bname]
    lost = tried = heard = shown = 0
    for seed in range(N):
        rng = np.random.default_rng(1000 + seed)
        L = rng.uniform(*map(float, os.environ["LEN"].split(","))) if "LEN" in os.environ else (rng.uniform(36.0, 39.5) if mx == 30.0 else rng.uniform(126.0, 129.5))
        # start the stretch at a word start in the squeezed chapter
        i = int(rng.integers(0, len(sqw) - 400))
        s0 = int(sqw[i, 0]) - int(0.05 * SR)
        speech = sq[s0: s0 + int(L * SR)].copy()
        # end it at a word end: zero after the last word end inside
        inside = sqw[(sqw[:, 1] > s0) & (sqw[:, 1] < s0 + speech.size)]
        last_end = int(inside[-1, 1]) - s0 + int(0.05 * SR)
        speech[last_end:] = 0
        speech *= 10 ** ((float(os.environ.get('SPEECH_DB', '-20')) - rms_db(speech[:last_end])) / 20)
        lead = 3.0 if side == "lead" else 0.5
        tail = 3.0 if side == "tail" else 0.3
        total = int((lead + tail) * SR) + speech.size
        wav = np.zeros(total, np.float32)
        a0 = int(lead * SR)
        wav[a0: a0 + speech.size] = speech
        wa, wb = long_words[int(rng.integers(0, len(long_words)))]
        word = clean[wa:wb].astype(np.float64)
        word -= word.mean()
        level = (bed_db + rel) if kind != "none" else rel
        word = word / np.sqrt(np.mean(word ** 2)) * 10 ** (level / 20)
        if side == "tail":
            at = a0 + last_end + int(rng.uniform(1.2, 2.2) * SR)
        else:
            at = a0 - word.size - int(rng.uniform(1.2, 2.2) * SR)
        wav[at: at + word.size] += word.astype(np.float32)
        wav += bed(kind, total, bed_db, rng) if kind != "none" else (rng.standard_normal(total) * 10 ** (-65 / 20)).astype(np.float32)
        a, b = at, at + word.size
        res = {}
        for name, mod in (("main", MAIN), ("pr", PR)):
            p = mod.plan_chunks(wav, target_sec=target, max_sec=mx, min_sec=20.0, context_sec=ctx)
            res[name] = dict(heard=any(s < b and e > a for s, e in p.speech),
                             inrange=any(r0 <= a and b <= r1 for r0, r1 in p.ranges),
                             inwin=any(w0 <= a and b <= w1 for w0, w1 in p.windows),
                             ranges=[(round(r0 / SR, 2), round(r1 / SR, 2)) for r0, r1 in p.ranges],
                             speech=[(round(s / SR, 2), round(e / SR, 2)) for s, e in p.speech])
        tried += 1
        heard += res["main"]["heard"]
        if res["main"]["inrange"] and not res["pr"]["inwin"]:
            lost += 1
            if shown < 2:
                shown += 1
                print(f"  seed {seed}: word {a/SR:.2f}-{b/SR:.2f}s at {level:.0f} dBFS ({kind} bed {bed_db:.0f} dBFS); speech {rms_db(speech[:last_end]):.1f} dBFS; VAD {res['main']['speech']}\n"
                      f"    main ranges {res['main']['ranges']}\n    head ranges {res['pr']['ranges']}")
    print(f"real {side} bed={kind}@{bed_db} word {rel:+.0f} dB re bed, bounds {bname}: VAD heard the word {heard}/{tried}; word in main's range, outside every head window: {lost}/{tried}")


if __name__ == "__main__":
    main()
