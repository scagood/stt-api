import asyncio, logging
import lib
from lib import routes
logging.disable(logging.CRITICAL)
truth = lib.filler(0.5, 40, 0.5)
m = lib.Model(truth, lambda a, b, ph, i: dict(skip=[(20, 30)]) if (a, b) == (0.0, 40.0) else {})
class W:
    async def submit_many(self, pieces, key):
        return [m.decode(x[0] / 16000, x[0] / 16000 + x.size / 16000, "", 0) for x in pieces]
def boom(wav): raise MemoryError("vad")
routes.speech_segments = boom
p = lib.prepared(40, [(0, 40)], [(0, 40)], None)
async def go():
    first = await W().submit_many(p.pieces, "k")
    return await routes._redo_stalled(lib.request(W()), [p], first, "parakeet-v3:fp32")
try:
    r = asyncio.run(go()); print("VAD MemoryError -> kept", len(routes._stitch(p, r)[2]), "words")
except BaseException as e:
    print("VAD exception propagates:", repr(e))
routes._aligned = lambda *a: (_ for _ in ()).throw(ValueError("aligned"))
routes.speech_segments = lambda wav: [(0, wav.size)]
try:
    r = asyncio.run(go()); print("_aligned ValueError -> kept", len(routes._stitch(p, r)[2]), "words")
except BaseException as e:
    print("_aligned exception propagates:", repr(e))
