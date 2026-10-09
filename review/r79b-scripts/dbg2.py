import os, sys
sys.argv = ["fuzz2.py"] + sys.argv[1:]
import lib
from lib import routes
LO, HI = float(os.environ["LO"]), float(os.environ["HI"])
def toks(result, origin_s):
    info = routes._extract(result)
    return [(t, round(origin_s + s, 3)) for t, s in zip(info["tokens"], info["timestamps"]) if LO <= origin_s + s <= HI]
orig_m = routes._merged
def traced(result, origin, again, start, stop, taken=()):
    out, added = orig_m(result, origin, again, start, stop, taken)
    print("MERGED window", start / 16000, stop / 16000)
    print("  piece :", toks(result, origin / 16000))
    print("  redo  :", toks(again, start / 16000))
    print("  out   :", toks(out, origin / 16000))
    return out, added
routes._merged = traced
exec(open("fuzz2.py").read())
