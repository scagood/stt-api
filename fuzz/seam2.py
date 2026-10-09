import sys, re
argv = sys.orig_argv
seed, kind, lo, hi = int(argv[-4]), argv[-3], int(argv[-2]), int(argv[-1])
sys.argv = ["x"]
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import fuzz
fuzz.REPEAT, fuzz.SPLIT, fuzz.MADEUP, fuzz.DRIFT = 0.05, 0.1, 0.5, 0.3
routes = fuzz.VERSIONS["parakeet_service"]
pick = lambda w: (m := re.fullmatch(r"[wvsu](\d+)x*[ab]?", w)) and lo <= int(m.group(1)) <= hi
orig_trim = routes._trimmed
def trim(prepared, results):
    for k, (r, w) in enumerate(zip(results, prepared.windows)):
        info = routes._extract(r)
        print(f"piece {k} window {w[0]/16000:.2f}-{w[1]/16000:.2f} range {prepared.ranges[k][0]/16000:.2f}-{prepared.ranges[k][1]/16000:.2f}:",
              [(x, round(w[0] / 16000 + info["timestamps"][f], 2)) for x, f, _l in routes._word_spans(info["tokens"]) if pick(x)])
    out = orig_trim(prepared, results)
    for k, r in enumerate(out):
        print("  kept", k, [x for x, f, _l in routes._word_spans(routes._extract(r)["tokens"]) if pick(x)])
    return out
routes._trimmed = trim
real = routes._heard_near
def kept(*a):
    out = real(*a); print("taken near cut", round(a[3] / 16000, 2), [(round(x / 16000, 2), k) for x, _y, k in out]); return out
routes._heard_near = kept
c = fuzz.case(seed, kind, 0.1)
fuzz.run(routes, c)
