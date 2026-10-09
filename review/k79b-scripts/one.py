import sys, random, json, logging
sys.argv = [sys.argv[0], sys.argv[1]] + sys.argv[2:]
import sim
logging.disable(logging.CRITICAL)
seed, case, jitter, mode = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
rng = random.Random(seed * 100003 + case)
# replicate fuzz.py generation for mode two/three
if mode == "two":
    total = 120.0; cut = rng.uniform(50, 62); ctx = 5.0
    ranges = [(0.0, cut), (cut, total)]; windows = [(0.0, cut + ctx), (cut - ctx, total)]
else:
    total = 180.0; c1, c2 = rng.uniform(50, 62), rng.uniform(110, 122); ctx = 5.0
    ranges = [(0.0, c1), (c1, c2), (c2, total)]; windows = [(0.0, c1 + ctx), (c1 - ctx, c2 + ctx), (c2 - ctx, total)]
speech = [(0.0, total)]
truth, t, i = [], rng.uniform(0.0, 0.5), 0
while t < total - 0.2:
    truth.append((f"w{i}", round(t, 4))); i += 1; t += rng.uniform(0.25, 0.7)
print("ranges", ranges, "windows", windows)
