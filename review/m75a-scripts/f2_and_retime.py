import sys
sys.path.insert(0, ".")
import numpy as np
from parakeet_service import retime, chunker

sys.path.insert(0, sys.argv[1])
import repro  # noqa: builds wavs (runs its cases unless filtered)


def w(t, a, b):
    return {"word": t, "start": a, "end": b}


words = [w("Then", 10.0, 11.8), w("said", 13.0, 13.4), w("him.", 14.0, 14.4), w("Next", 18.5, 19.0)]
starts = [x["start"] for x in words]
print("crowded:", retime._crowded((12.0, 18.0), words, starts, 0.0, 30.0))
out = retime.retime(words, [(12.0, 18.0)], 0.0, 30.0)
print("crossing:", [(x["word"], round(x["start"], 2), round(x["end"], 2)) for x in out])
words2 = [w("Then", 10.0, 11.8), w("him.", 13.0, 13.4), w("said", 14.0, 14.4), w("Next", 18.5, 19.0)]
out2 = retime.retime(words2, [(12.0, 18.0)], 0.0, 30.0)
print("legit order:", [(x["word"], round(x["start"], 2), round(x["end"], 2)) for x in out2])

# retime pauses on the multi-breath wav
wav = repro.build(15, [(43, 0.25, -42), (48, 0.25, -42), (53, 0.25, -42)])
ps = [(round(a, 2), round(b, 2)) for a, b in retime.pauses(wav) if b > 39 and a < 56]
print("retime pauses near 15s pause, 3 breaths:", ps)
wav = repro.build(15, [])
ps = [(round(a, 2), round(b, 2)) for a, b in retime.pauses(wav) if b > 39 and a < 56]
print("retime pauses near 15s pause, no breaths:", ps)
