import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from parakeet_service import chunker
SR = 16000
def at(*s): return tuple(int(x*SR) for x in s)
def show(name, speech, secs, t=25.0, mx=30.0, ctx=5.0, mn=20.0):
    chunker._speech_segments = lambda _w: speech
    r = chunker.auto_chunk(np.zeros(int(secs*SR), np.float32), target_sec=t, max_sec=mx, min_sec=mn, context_sec=ctx)
    inside = [e/SR for s, e in r[:-1] for a, b in speech if a < e < b]
    silent = [(s/SR, e/SR) for s, e in r if not any(a < e and s < b for a, b in speech)]
    print(f"{name}: ranges={[(s/SR, e/SR) for s, e in r]} cuts_inside_speech={inside} silent={silent}")
show("A next phrase fits only from late in the pause", [at(3, 13), at(14, 33.8)], 40)
show("B first phrase fits but leading margin pushes it over", [at(2, 21.5), at(22.5, 30)], 40)
show("B' whisper", [at(1, 29.5), at(30.5, 40)], 50, ctx=0.0)
show("C silent sliver mid", [at(3, 40), at(41, 50)], 60)
show("C' silent sliver last", [at(3, 40)], 50)
show("D 1 s phrase then 19 s phrase", [at(3, 4), at(4.6, 23.6), at(24.2, 30)], 40)
