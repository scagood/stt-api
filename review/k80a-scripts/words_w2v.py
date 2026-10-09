"""Word times for a 16 kHz f32 file by wav2vec2 (the repo's aligner):
greedy CTC transcript, then forced alignment (aligner.word_spans) in ~20 s groups.
usage: words_w2v.py IN.f32 OUT.npy"""
import sys, time
sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/k80a-pr")
import numpy as np
from parakeet_service import aligner
from parakeet_service.config import ALIGNER_CONFIGS

src, dst = sys.argv[1], sys.argv[2]
wav = np.fromfile(src, dtype=np.float32)
name = "wav2vec2-base-960h"
spec = ALIGNER_CONFIGS[name]
t0 = time.time()
session, vocab = aligner._load(name, spec["default_quantization"])
print("loaded", time.time() - t0, flush=True)
t0 = time.time()
em, fs = aligner._emission(session, wav)
print("emission", em.shape, time.time() - t0, flush=True)
inv = {v: k for k, v in vocab.items()}
blank, sep = vocab[spec["blank"]], vocab[spec["separator"]]
ids = em.argmax(axis=1)
words = []  # (letters, first frame, last frame)
cur, first, last, prev = [], None, None, None
for t, i in enumerate(ids.tolist()):
    if i == prev:
        if i not in (blank, sep) and cur:
            last = t
        continue
    prev = i
    if i == blank:
        continue
    if i == sep:
        if cur:
            words.append(("".join(cur), first, last)); cur = []
        continue
    ch = inv[i]
    if len(ch) != 1 or not (ch.isalpha() or ch == "'"):
        continue
    if not cur:
        first = t
    cur.append(ch); last = t
if cur:
    words.append(("".join(cur), first, last))
print("greedy words", len(words), flush=True)
spans = []
group = []
def flush(group):
    lo = max(0, group[0][1] - 25); hi = min(em.shape[0], group[-1][2] + 26)
    spoken = [(k, [vocab[c] for c in w if c in vocab]) for k, (w, a, b) in enumerate(group)]
    timed = aligner.word_spans(em[lo:hi], fs[lo:hi], spoken, len(group), blank=blank, separator=sep)
    if timed is None:
        print("align failed for group at", fs[lo]); return
    for (w, a, b), span in zip(group, timed):
        if span is not None:
            spans.append((span[0], span[1]))
for w in words:
    if group and (w[1] - group[0][1]) > 1000 and w[1] - group[-1][2] > 5:  # ~20 s, split at a gap of 100 ms+
        flush(group); group = []
    group.append(w)
if group:
    flush(group)
spans = np.array(sorted(spans))
np.save(dst, spans)
print("aligned words", len(spans), "speech-in-words share", float((spans[:, 1] - spans[:, 0]).sum() / (wav.size / 16000)), time.time() - t0, flush=True)
