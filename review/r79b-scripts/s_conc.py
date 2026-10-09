import asyncio, logging, random, concurrent.futures, json
from types import SimpleNamespace
import numpy as np
import lib
from lib import routes, S, SR
logging.disable(logging.CRITICAL)
def make(k):
    rng = random.Random(k)
    pieces = rng.choice([1, 2, 3])
    if pieces == 1:
        total = rng.uniform(20, 75); ranges = windows = [(0, total)]; speech = None
    elif pieces == 2:
        total = 120.0; ranges, windows = lib.two(total, rng.uniform(50, 70))
        speech = [(0, total)]
    else:
        total = 180.0; ranges, windows = lib.three(); speech = [(0, total)]
    truth = lib.filler(rng.uniform(0, 0.4), total, rng.uniform(0.3, 0.6), f"f{k}w")
    first = {tuple(np.round(w, 3)): [(rng.uniform(w[0], w[1] - 5), 0)] for w in windows}
    for key in first:
        a = first[key][0][0]; first[key] = [(a, a + rng.uniform(4, 12))]
    range_skip = rng.random() < 0.5
    return total, ranges, windows, speech, truth, first, range_skip
class Worker:
    def __init__(self, specs): self.specs = specs
    async def submit_many(self, pieces, key):
        await asyncio.sleep(0.001)
        out = []
        for p in pieces:
            # the piece's samples encode the clip id in the fractional part
            k = int(round((p[0] % 1) * 100)); a = float(np.floor(p[0])) / SR; b = a + p.size / SR
            total, ranges, windows, speech, truth, first, range_skip = self.specs[k]
            fk = (round(a, 3), round(b, 3))
            skip = first.get(fk, [])
            if not skip and range_skip and any(abs(a - r[0]) < 1e-3 and abs(b - r[1]) < 1e-3 for r in ranges):
                skip = list(first.values())[0]
            m = lib.Model(truth, lambda *_: dict(skip=skip))
            out.append(m.decode(a, b, "x", 0))
        return out
def prep(k, spec):
    total, ranges, windows, speech, truth, first, range_skip = spec
    p = lib.prepared(total, ranges, windows, speech)
    wav = np.arange(S(total), dtype=np.float64) + k / 100.0
    p.waveform = wav; p.pieces = [wav[S(a):S(b)] for a, b in windows]
    return p
specs = [make(k) for k in range(12)]
routes.speech_segments = lambda wav: [(0, wav.size)]
async def one(k, req):
    p = prep(k, specs[k])
    first = await req.app.state.worker.submit_many(p.pieces, "parakeet-v3:fp32")
    res = await routes._redo_stalled(req, [p], first, "parakeet-v3:fp32")
    text, segs, words = routes._stitch(p, res)
    return text
async def main():
    pool = concurrent.futures.ThreadPoolExecutor(4)
    req = lib.request(Worker(specs), pool)
    seq = [await one(k, req) for k in range(12)]
    for rep in range(5):
        con = await asyncio.gather(*(one(k, req) for k in range(12)))
        assert con == seq, "concurrent results differ"
    # a batch of all one-piece files together
    ones = [k for k in range(12) if len(specs[k][1]) == 1]
    ps = [prep(k, specs[k]) for k in ones]
    flat = await req.app.state.worker.submit_many([x for p in ps for x in p.pieces], "parakeet-v3:fp32")
    res = await routes._redo_stalled(req, ps, flat, "parakeet-v3:fp32")
    batch = [routes._stitch(p, [r])[0] for p, r in zip(ps, res)]
    print("batch == single:", batch == [seq[k] for k in ones], len(ones), "one-piece files")
    print("concurrent == sequential over 5 rounds of 12 requests; words", sum(len(t.split()) for t in seq))
asyncio.run(main())
