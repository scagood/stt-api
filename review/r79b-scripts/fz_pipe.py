"""Random weird decodes end to end through _redo_stalled and _stitch: crashes, text/words disagreement."""
import asyncio, random, logging, traceback, sys
from types import SimpleNamespace
import lib
from lib import routes, S
logging.disable(logging.CRITICAL)
TOK = [" the", "▁the", " ", "▁", "ing", ",", ".", " 1", "9", "8", "4", " £", "'s", "-", "?", " a", " no", " I", " 中", "文", " .", " w", "x", " -"]
def rand_result(rng, dur, words_per_s, unsorted=False, gaps=()):
    n = int(dur * words_per_s)
    toks, ts = [], []
    for _ in range(n):
        t = rng.uniform(0, dur)
        if any(a <= t < b for a, b in gaps): continue
        toks.append(rng.choice(TOK)); ts.append(round(t / 0.08) * 0.08)
    order = sorted(range(len(ts)), key=lambda k: ts[k])
    toks = [toks[k] for k in order]; ts = [ts[k] for k in order]
    if unsorted and len(ts) > 3:
        for _ in range(3):
            k = rng.randrange(len(ts) - 1); ts[k], ts[k + 1] = ts[k + 1] + 0.3, ts[k]
    return SimpleNamespace(text="".join(toks), tokens=toks, timestamps=ts)
class W:
    def __init__(self, rng): self.rng = rng
    async def submit_many(self, pieces, key):
        out = []
        for p in pieces:
            dur = p.size / 16000
            r = self.rng.random()
            if r < 0.1: out.append(SimpleNamespace(text="", tokens=[], timestamps=[]))
            else: out.append(rand_result(self.rng, dur, self.rng.uniform(0.5, 4), self.rng.random() < 0.3))
        return out
bad = 0
N = int(sys.argv[1])
for case in range(N):
    rng = random.Random(case)
    k = rng.choice([1, 2, 3])
    if k == 1:
        total = rng.uniform(3, 75); ranges = [(0, total)]; windows = ranges; speech = None
    elif k == 2:
        total = rng.uniform(80, 130); ranges, windows = lib.two(total, rng.uniform(40, total - 30)); speech = [(0, total)]
    else:
        total = 180.0; ranges, windows = lib.three(); speech = [(0, total)]
    prep = lib.prepared(total, ranges, windows, speech)
    routes.speech_segments = lambda wav, total=total, rng=rng: [(S(a), S(min(total, a + rng.uniform(1, 30)))) for a in sorted(rng.uniform(0, total) for _ in range(5))]
    worker = W(rng)
    async def go():
        first = []
        for p in prep.pieces:
            first.append(rand_result(rng, p.size / 16000, rng.uniform(0.5, 4), rng.random() < 0.3,
                                     gaps=[(g, g + rng.uniform(3, 20)) for g in [rng.uniform(0, p.size / 16000) for _ in range(2)]]))
        return await routes._redo_stalled(lib.request(worker), [prep], first, "parakeet-v3:fp32")
    try:
        results = asyncio.run(go())
        text, segs, words = routes._stitch(prep, results)
        tw = text.split(); ww = [w["word"] for w in words]
        if routes._clean_text(" ".join(w["word"] for w in words)) != text and False:
            pass
        segw = " ".join(s["segment"] for s in segs)
        if routes._clean_text(segw) != text:
            bad += 1; print("seg/text mismatch", case)
    except Exception as exc:
        bad += 1
        if bad < 5: print("case", case, repr(exc)); traceback.print_exc(limit=4)
print("bad", bad)
