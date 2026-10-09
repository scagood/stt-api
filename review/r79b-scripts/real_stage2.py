"""Stage 2: run head's real _redo_stalled/_stitch on the real decodes from stage 1."""
import asyncio, json, logging, sys
from collections import Counter
from types import SimpleNamespace
import lib
from lib import routes, SR
logging.disable(logging.CRITICAL)
CLIP = 30.0
def holed(toks, ts, s, e):
    spans = routes._word_spans(toks)
    drop = set()
    prev = -1
    for word, first, last in spans:
        if s <= ts[first] < e:
            drop.update(range(prev + 1, last + 1))
        prev = last
    keep = [k for k in range(len(toks)) if k not in drop]
    return SimpleNamespace(text=routes._decoded([toks[k] for k in keep]), tokens=[toks[k] for k in keep], timestamps=[ts[k] for k in keep])
def words(toks, ts, origin=0.0):
    return [(w, round(origin + ts[f], 3), round(origin + ts[l], 3)) for w, f, l in routes._word_spans(toks)]
tot = Counter()
verbose = len(sys.argv) > 2
for line in open(sys.argv[1]):
    rec = json.loads(line)
    P = words(rec["ptoks"], rec["pts"])
    for h in rec["holes"]:
        piece = holed(rec["ptoks"], rec["pts"], h["s"], h["e"])
        R = SimpleNamespace(text="", tokens=h["rtoks"], timestamps=h["rts"])
        asked = []
        class W:
            async def submit_many(self, pieces, key):
                asked.extend((x[0] / SR, (x[0] + x.size) / SR) for x in pieces)
                return [R if abs(x[0] / SR - h["a"]) < 0.01 and abs((x[0] + x.size) / SR - h["b"]) < 0.01 else SimpleNamespace(text="", tokens=[], timestamps=[]) for x in pieces]
        prep = lib.prepared(CLIP, [(0, CLIP)], [(0, CLIP)], None)
        routes.speech_segments = lambda wav, h=h: [(int(h["s"] * SR), int(h["e"] * SR))]
        res = asyncio.run(routes._redo_stalled(lib.request(W()), [prep], [piece], "parakeet-v3:fp32"))
        text, segs, out = routes._stitch(prep, res)
        if not asked:
            tot["no_redo"] += 1; continue
        match = [w for w in asked if abs(w[0] - h["a"]) < 0.01 and abs(w[1] - h["b"]) < 0.01]
        if len(asked) > 1: tot["extra_windows"] += len(asked) - 1
        a, b = match[0] if match else asked[0]
        if abs(a - h["a"]) > 0.01 or abs(b - h["b"]) > 0.01:
            tot["window_mismatch"] += 1; print("MISMATCH asked", (round(a,3), round(b,3)), "saved", (round(h["a"],3), round(h["b"],3)), "stretch", h["low"], h["high"]); continue
        tot["redos"] += 1
        main_words = Counter((w, round(t, 2)) for w, t, _l in words(piece.tokens, piece.timestamps))
        head_words = Counter((x["word"], round(x["start"], 2)) for x in out)
        # head words beyond the piece's own: inserted
        inserted = sorted((head_words - main_words).elements(), key=lambda x: x[1])
        lost = main_words - head_words
        tot["lost_piece_words"] += sum(lost.values())
        stretch_lo, stretch_hi = h["low"], h["high"]
        kept = len(res[0].tokens) != len(piece.tokens)
        tot["kept"] += kept
        hole_words = [x for x in P if h["s"] <= x[1] < h["e"]]
        tot["hole_words"] += len(hole_words)
        lo_, hi_ = min(stretch_lo, h["s"]) - 0.08, max(stretch_hi, h["e"])
        ins_in = [x for x in inserted if lo_ <= x[1] < hi_]
        ins_margin = [x for x in inserted if not (lo_ <= x[1] < hi_)]
        tot["inserted_in_stretch"] += len(ins_in)
        tot["inserted_in_margin"] += len(ins_margin)
        for word, at in ins_margin:
            ctxP = [(w, t) for w, t, _l in P if abs(t - at) <= 0.8]
            ctxR = [(w, round(t, 2)) for w, t, _l in words(h["rtoks"], h["rts"], a) if abs(t - at) <= 0.8]
            side = "before" if at < stretch_lo else "after"
            print(f"MARGIN {rec['name']}@{rec['off']:.1f} hole {h['s']:.2f}-{h['e']:.2f} stretch {stretch_lo:.2f}-{stretch_hi:.2f} redo {a:.2f}-{b:.2f}: inserted {word!r}@{at} ({side})")
            print("     piece:", ctxP)
            print("     redo :", ctxR)
        # inside the stretch: compare with what the full decode heard there
        keysP = Counter(routes._seam_key(w) for w, t, _l in P if h["s"] <= t < h["e"])
        keysI = Counter(routes._seam_key(w) for w, t in ins_in)
        tot["stretch_extra_vs_full_decode"] += sum((keysI - keysP).values())
        tot["stretch_missing_vs_full_decode"] += sum((keysP - keysI).values())
        if verbose and (keysI - keysP):
            print("STRETCH extra", rec["name"], round(rec["off"], 1), dict(keysI - keysP), "missing", dict(keysP - keysI))
print(dict(tot))
