import sys, re
argv = sys.orig_argv
seed, kind, target = int(argv[-3]), argv[-2], int(argv[-1])
sys.argv = ["x"]
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import fuzz
fuzz.REPEAT, fuzz.SPLIT, fuzz.MADEUP, fuzz.DRIFT = 0.05, 0.1, 0.5, 0.3
routes = fuzz.VERSIONS["parakeet_service"]
orig = routes._merged
def near(info, origin):
    return [(w, round(origin / 16000 + info["timestamps"][f], 2), round(origin / 16000 + info["timestamps"][l], 2)) for w, f, l in routes._word_spans(info["tokens"])
            if (m := re.fullmatch(r"[wvsu](\d+)x*[ab]?", w)) and abs(int(m.group(1)) - target) <= 2]
def traced(result, origin, again, start, stop, stretches, own, taken=()):
    global routes_taken; routes_taken = taken
    out, added = orig(result, origin, again, start, stop, stretches, own, taken)
    print("redo", round(start / 16000, 2), round(stop / 16000, 2), "own", [round(x / 16000, 2) for x in own], "stretches", [(round(a / 16000, 2), round(b / 16000, 2)) for a, b in stretches])
    print("   piece", near(routes._extract(result), origin)); print("   redo ", near(routes._extract(again), start)); print("   out  ", near(routes._extract(out), origin))
    return out, added
routes._merged = traced
c = fuzz.case(seed, kind, 0.1)
print("truth", [(fuzz.spelling(i), round(a, 2)) for i, a, _o in c.words if abs(abs(i) - target) <= 2 or abs(-i - 1 - target) <= 2])
said = fuzz.run(routes, c)[0]
print([w for w in said if (m := re.fullmatch(r"w(\d+)", w)) and abs(int(m.group(1)) - target) <= 2])
