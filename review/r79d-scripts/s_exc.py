import asyncio, logging
import lib
from lib import routes
logging.disable(logging.CRITICAL)
truth = lib.filler(0.5, 40, 0.5)
class W:
    def __init__(self, exc): self.exc, self.n = exc, 0
    async def submit_many(self, pieces, key):
        self.n += 1
        if self.n > 1: raise self.exc
        m = lib.Model(truth, lambda *a: dict(skip=[(20, 30)]))
        return [m.decode(0, 40, "first", 0)]
for exc in (RuntimeError("worker died"), routes.ModelLoadError("evicted") if hasattr(routes, "ModelLoadError") else RuntimeError(), asyncio.CancelledError()):
    p = lib.prepared(40, [(0, 40)], [(0, 40)], None)
    routes.speech_segments = lambda wav: [(0, wav.size)]
    w = W(exc)
    async def go():
        first = await w.submit_many(p.pieces, "k")
        return await routes._redo_stalled(lib.request(w), [p], first, "parakeet-v3:fp32")
    try:
        r = asyncio.run(go()); print(type(exc).__name__, "-> kept first decode,", len(routes._stitch(p, r)[2]), "words")
    except BaseException as e:
        print(type(exc).__name__, "-> propagated", type(e).__name__)
# an exception inside the splice itself
orig = routes._merged
def boom(*a, **k): raise IndexError("in _merged")
routes._merged = boom
w = W(None); w.n = -10
p = lib.prepared(40, [(0, 40)], [(0, 40)], None)
async def go():
    m = lib.Model(truth, lambda a, b, ph, i: dict(skip=[(20, 30)]) if a == 0 and b == 40 else {})
    class W2:
        async def submit_many(self, pieces, key):
            return [m.decode(x[0] / 16000, x[0] / 16000 + x.size / 16000, "", 0) for x in pieces]
    first = await W2().submit_many(p.pieces, "k")
    return await routes._redo_stalled(lib.request(W2()), [p], first, "parakeet-v3:fp32")
try:
    asyncio.run(go()); print("splice exception swallowed")
except Exception as e:
    print("exception in _merged -> propagates out of _redo_stalled:", repr(e))
