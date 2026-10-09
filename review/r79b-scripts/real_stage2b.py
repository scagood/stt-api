"""Stage 2b: _merged of each real redo into the FULL piece decode (nothing skipped):
anything put in is speech the piece already had, heard again (duplicate) or made up."""
import json, sys
from collections import Counter
from types import SimpleNamespace
import lib
from lib import routes, SR
tot = Counter()
for line in open(sys.argv[1]):
    rec = json.loads(line)
    P = SimpleNamespace(text="", tokens=rec["ptoks"], timestamps=rec["pts"])
    Pw = [(w, round(rec["pts"][f], 2)) for w, f, l in routes._word_spans(rec["ptoks"])]
    for h in rec["holes"]:
        R = SimpleNamespace(text="", tokens=h["rtoks"], timestamps=h["rts"])
        a, b = int(round(h["a"] * SR)), int(round(h["b"] * SR))
        out, added = routes._merged(P, 0, R, a, b, ())
        tot["redos"] += 1
        tot["redo_words"] += len(routes._word_spans(h["rtoks"]))
        tot["added"] += len(added)
        if added:
            tot["redos_with_added"] += 1
            Rw = [(w, round(h["a"] + h["rts"][f], 2)) for w, f, l in routes._word_spans(h["rtoks"])]
            for at in added:
                t = at / SR
                if not [x for x in Pw if abs(x[1] - t) <= 1.0]:
                    tot["added_in_piece_gap"] += 1
                    continue
                tot["added_near_piece_words"] += 1
                print(f"{rec['name']}@{rec['off']:.1f} redo {h['a']:.2f}-{h['b']:.2f}: put in at {t:.2f} ({t - h['a']:.2f} s after its start, {h['b'] - t:.2f} s before its end)")
                print("    piece:", [x for x in Pw if abs(x[1] - t) <= 1.0])
                print("    redo :", [x for x in Rw if abs(x[1] - t) <= 1.0])
print(dict(tot))
