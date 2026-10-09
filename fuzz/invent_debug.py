import sys
argv = sys.orig_argv
seed, kind = int(argv[-2]), argv[-1]
sys.argv = ["x"]
sys.path.insert(0, __file__.rsplit("/", 1)[0]); import fuzz
fuzz.REPEAT, fuzz.SPLIT, fuzz.MADEUP, fuzz.DRIFT = 0.03, 0.05, 0.3, 0.2
routes = fuzz.VERSIONS["parakeet_service"]
orig = routes._merged
def traced(result, origin, again, start, stop, stretches, own, taken=()):
    out, added = orig(result, origin, again, start, stop, stretches, own, taken)
    info = routes._extract(again)
    made = [(w, round(start / 16000 + info["timestamps"][f], 2)) for w, f, _l in routes._word_spans(info["tokens"]) if w.startswith("u")]
    if made:
        pinfo = routes._extract(result)
        print("redo", round(start / 16000, 2), round(stop / 16000, 2), "own", [round(x / 16000, 2) for x in own], "stretches", [(round(a / 16000, 2), round(b / 16000, 2)) for a, b in stretches])
        print("  made up", made, "added", [round(a / 16000, 2) for a in added])
        print("  piece words near end:", [(w, round(origin / 16000 + pinfo["timestamps"][f], 2)) for w, f, _l in routes._word_spans(pinfo["tokens"]) if abs(origin / 16000 + pinfo["timestamps"][f] - made[0][1]) < 2.5])
        print("  redo words near end:", [(w, round(start / 16000 + info["timestamps"][f], 2)) for w, f, _l in routes._word_spans(info["tokens"]) if abs(start / 16000 + info["timestamps"][f] - made[0][1]) < 2.5])
    return out, added
routes._merged = traced
c = fuzz.case(seed, kind, 0.07)
print(fuzz.run(routes, c)[0][-6:], "duration", c.wav.size / 16000)
