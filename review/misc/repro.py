import sys, importlib.util, types
sys.path.insert(0, sys.argv[1])
import numpy as np
import parakeet_service.chunker as c
SR = c.TARGET_SR
def run(speech, total, **b):
    c._speech_segments = lambda w: [(int(a*SR), int(e*SR)) for a, e in speech]
    r = c.plan_chunks(np.zeros(int(total*SR), dtype=np.float32), **b).ranges
    return [(round(a/SR,2), round(e/SR,2)) for a, e in r]
v2 = dict(target_sec=25.0, max_sec=30.0, min_sec=20.0, context_sec=5.0)
print(run([(0,18.8),(21.5,43.5)], 43.5, **v2))
print(run([(3,10),(10.5,41.5)], 45, **v2))
print(run([(3,41.5)], 45, **v2))
print(run([(0,100)], 100, target_sec=60.0, max_sec=75.0, min_sec=20.0, context_sec=5.0))
