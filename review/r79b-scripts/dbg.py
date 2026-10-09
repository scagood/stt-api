"""Re-run a fuzz2 case with tracing of _merged inputs/outputs and _seam."""
import os, sys, json
sys.argv = ["fuzz2.py"] + sys.argv[1:]
import lib
from lib import routes
SUF = "~^*+=<>|"
def ident(word):
    base = word.rstrip(SUF); suf = word[len(base):]
    if not suf: return None
    n = 0
    for ch in suf: n = n * 8 + SUF.index(ch)
    return n
LO, HI = float(os.environ["LO"]), float(os.environ["HI"])
def words_of(result, origin_s):
    info = routes._extract(result)
    return [(w, round(origin_s + info["timestamps"][f], 3)) for w, f, l in routes._word_spans(info["tokens"]) if LO <= origin_s + info["timestamps"][f] <= HI]
if hasattr(routes, "_merged"):
    orig_m = routes._merged
    def traced(result, origin, again, start, stop, taken=()):
        out, added = orig_m(result, origin, again, start, stop, taken)
        print("MERGED window", start / 16000, stop / 16000, "origin", origin / 16000)
        print("  piece :", words_of(result, origin / 16000))
        print("  redo  :", words_of(again, start / 16000))
        print("  out   :", words_of(out, origin / 16000))
        print("  taken :", [(a / 16000, b / 16000) for a, b in taken])
        return out, added
    routes._merged = traced
orig_seam = routes._seam
def seam(left, right, cut, reach):
    r = orig_seam(left, right, cut, reach)
    if LO <= cut <= HI:
        print("SEAM cut", cut, "reach", reach)
        print("  left :", [(w, round(t, 3), i) for i, (w, t) in enumerate(left) if abs(t - cut) <= reach])
        print("  right:", [(w, round(t, 3), j) for j, (w, t) in enumerate(right) if abs(t - cut) <= reach])
        print("  ->", r)
    return r
routes._seam = seam
exec(open("fuzz2.py").read())
