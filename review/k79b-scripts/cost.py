import asyncio, sys, logging
sys.argv = [sys.argv[0], sys.argv[1]]
import sim
logging.basicConfig(level=logging.WARNING, format="%(message)s")
# 120 s, two pieces; the first decode hears one word every 3.5 s (lyrics over music, VAD: speech throughout)
truth = [(f"w{i}", 0.2 + 3.5 * i) for i in range(34)]
text, segs, words, calls = sim.run(truth, [(0.0, 60.0), (60.0, 120.0)], [(0.0, 65.0), (55.0, 120.0)], [(0.0, 120.0)], 120.0, [[], []])
print("redo decodes:", len(calls), "seconds:", round(sum(b - a for a, b in calls), 1))
