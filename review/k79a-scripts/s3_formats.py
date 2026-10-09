"""Spliced words through every response format, speak, align and retime paths."""
import asyncio
import sys

sys.path.insert(0, sys.argv[1])
sys.path.insert(1, sys.argv[2])
import simlib  # noqa: E402
from simlib import Model, run, words_every, S, SR  # noqa: E402
from parakeet_service import aligner, routes  # noqa: E402

aligner.ALIGN_DEFAULT_LANGUAGE = "en"
heard_audio = []


class FakeChunk:
    def __init__(self, wav):
        self.wav = wav

    def spans(self, words):
        heard_audio.append((float(self.wav[0]) / SR, self.wav.size / SR, list(words)))
        # keep each word near where it is: evenly over the audio
        n = len(words)
        dur = self.wav.size / SR
        return [(dur * i / n, dur * (i + 0.5) / n) for i in range(n)]

    def scores(self, options, _s, _e):
        return [0.0] * len(options)

    def best(self, _o, _s, _e):
        return 0


aligner.for_chunk = lambda wav, language, name=None, quantization=None: FakeChunk(wav)

opening = [([" Nineteen"], [0.5]), ([" eighty", "-", "four", ","], [1.2, 1.5, 1.7, 1.9]),
           ([" by"], [2.4]), ([" George"], [2.8]), ([" Or", "well", "."], [3.3, 3.5, 3.8]),
           ([" It"], [5.0]), ([" was"], [5.4]), ([" ", "1", "9", "8", "4", "."], [5.7, 5.8, 5.9, 6.0, 6.1, 6.2])]
rest = [([" Part"], [10.4]), ([" one", "."], [10.8, 11.0])] + words_every(12.0, 69.5, 1.0)
truth = opening + rest


def check(name, out, speak=False):
    words = out.words
    text = out.text
    seg_text = " ".join(s["segment"] for s in out.segments)
    word_text = " ".join(w["word"] for w in words) if words is not None else None
    starts = [w["start"] for w in words]
    mono = all(a <= b + 1e-9 for a, b in zip(starts, starts[1:]))
    srt = routes._segments_to_srt(out.segments)
    vtt = routes._segments_to_vtt(out.segments)
    vj = routes._verbose_json("en", out.prep.duration, text, out.segments, words)
    print(f"{name}: text[:80]={text[:80]!r}")
    print(f"   text==segments={text == routes._clean_text(seg_text)} text==words={text == word_text} monotonic={mono} "
          f"srt_has_opening={'ineteen' in srt} vtt_has_opening={'ineteen' in vtt} vj_words={len(vj.get('words') or [])} "
          f"first_words={[(w['word'], round(w['start'], 2)) for w in words[:4]]}")


one = Model(truth, skips=lambda a, b: [(0.0, 10.3)] if (a, b) == (0, 70) else [])
check("plain", run(70, [(0, 70)], [(0, 70)], None, one, vad=[(0, 70)]))
for kw in [dict(speak=True, language="en"), dict(align=True, language="en", aligner_choice=("wav2vec2-base-960h", "int8")),
           dict(retime_words=True)]:
    heard_audio.clear()
    one = Model(truth, skips=lambda a, b: [(0.0, 10.3)] if (a, b) == (0, 70) else [])
    out = run(70, [(0, 70)], [(0, 70)], None, one, vad=[(0, 70)], **kw)
    check(str(sorted(kw)), out)
    for a, size, words in heard_audio:
        print(f"   aligner heard {a:.2f}+{size:.2f}s, {len(words)} words, first {words[:3]}")

# multi-piece: piece 2 skips its opening; aligner audio for piece 2
heard_audio.clear()
truth2 = words_every(0.2, 59.5, 0.6) + opening_shift if False else None
truth2 = words_every(0.2, 59.5, 0.6) + [([w[0][0].replace(' ', ' ')] + w[0][1:], [t + 60.5 for t in w[1]]) for w in opening] \
    + [([" Part"], [70.9]), ([" one", "."], [71.3, 71.5])] + words_every(72.5, 120, 0.6, prefix="v")
m = Model(truth2, skips=lambda a, b: [(55.0, 70.8)] if (a, b) == (55, 120) else [])
out = run(120, [(0, 60), (60, 120)], [(0, 65), (55, 120)], [(0, 120)], m,
          align=True, language="en", aligner_choice=("wav2vec2-base-960h", "int8"))
check("multi align", out)
for a, size, words in heard_audio:
    print(f"   aligner heard {a:.2f}+{size:.2f}s, {len(words)} words, first {words[:3]}")
m = Model(truth2, skips=lambda a, b: [(55.0, 70.8)] if (a, b) == (55, 120) else [])
out = run(120, [(0, 60), (60, 120)], [(0, 65), (55, 120)], [(0, 120)], m, speak=True, language="en")
check("multi speak", out)
print("   around 60:", [(w["word"], round(w["start"], 2)) for w in out.words if 59 < w["start"] < 72])
