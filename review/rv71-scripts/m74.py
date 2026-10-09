import sys; sys.path.insert(0, ".")
import numpy as np
from parakeet_service import chunker
SR = 16000
def run(sp, sec):
    chunker._speech_segments = lambda _w: [tuple(int(x*SR) for x in p) for p in sp]
    r = chunker.plan_chunks(np.zeros(int(sec*SR), dtype=np.float32), target_sec=25.0, max_sec=30.0, min_sec=20.0, context_sec=5.0).ranges
    print(sp, [(a/SR, b/SR) for a, b in r])
run([(3,40)], 60); run([(3,40),(41,50)], 60); run([(3,13),(14,33.8)], 40)
