from h import *
import fuzz
fuzz.rng = np.random.default_rng(1)
rows = []
for _ in range(3000):
    sp, seconds = fuzz.layout()
    if not sp: continue
    spS = [tuple(int(x * SR) for x in s) for s in sp]; total = int(seconds * SR)
    p = plan(new, spS, total, BOUNDS["v2"])
    if not p.speech: continue
    r = p.ranges
    s, e = r[-1]
    if not any(a < e and s < z for a, z in p.speech) and len(r) > 1 and r[-2][1] == s:
        rows.append(((s - p.speech[-1][1]) / SR, (e - s) / SR))
import statistics
print(len(rows), "silent trailing pieces; margin already in previous piece (s): min %.2f median %.2f; silent piece length median %.2f, max %.2f" % (
    min(x for x, _ in rows), statistics.median(x for x, _ in rows), statistics.median(y for _, y in rows), max(y for _, y in rows)))
print(sum(1 for x, _ in rows if x >= 0.5), "with >= 0.5 s margin already kept")
print(run(new, [(3, 37.5)], 60), run(new, [(3, 39.9)], 60))
