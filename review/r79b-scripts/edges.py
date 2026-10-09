import json, sys
import lib
from lib import routes
for line in open(sys.argv[1]):
    rec = json.loads(line)
    Pw = [(w, round(rec["pts"][f], 2)) for w, f, l in routes._word_spans(rec["ptoks"])]
    for h in rec["holes"]:
        Rw = [(w, round(h["a"] + h["rts"][f], 2)) for w, f, l in routes._word_spans(h["rtoks"])]
        kind = h.get("kind", "rand")
        if kind in ("end", "rand"):
            print(f"END  {rec['off']:.1f} input ends {h['b']:.2f}: P", [x for x in Pw if h['b'] - 1.2 <= x[1] <= h['b'] + 0.6], "\n        R", [x for x in Rw if x[1] >= h['b'] - 1.2])
        if kind in ("start", "rand"):
            print(f"START {rec['off']:.1f} input starts {h['a']:.2f}: P", [x for x in Pw if h['a'] - 0.6 <= x[1] <= h['a'] + 1.2], "\n        R", [x for x in Rw if x[1] <= h['a'] + 1.2])
