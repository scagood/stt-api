import sys
from collections import Counter
sys.argv = ["x"]; sys.path.insert(0, __file__.rsplit("/", 1)[0]); import fuzz
fuzz.REPEAT, fuzz.SPLIT, fuzz.MADEUP, fuzz.DRIFT = 0.05, 0.1, 0.5, 0.3
for seed, kind in [(4918484, "one"), (5575350, "two"), (1132670, "three"), (5005061, "three"), (5614824, "three"), (6660132, "three")]:
    c = fuzz.case(seed, kind, 0.1)
    truth = Counter(fuzz.spelling(i) for i, _a, _o in c.words)
    m, p = (Counter(fuzz.run(fuzz.VERSIONS[n], c)[0]) for n in ("parakeet_main", "parakeet_service"))
    print(seed, kind, "dup", [w for w in p if w in truth and p[w] > max(m[w], truth[w])], "lost", sum(max(0, min(m[w], truth[w]) - p[w]) for w in truth),
          "invented", [w for w in p if w not in truth and p[w] > m[w]], "missing", sum(max(0, truth[w] - p[w]) for w in truth))
