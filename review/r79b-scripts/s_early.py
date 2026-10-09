import logging, asyncio
from types import SimpleNamespace
import lib
from lib import routes, S
logging.disable(logging.CRITICAL)
def heard(origin, words):
    return SimpleNamespace(text="".join(t for t, _ in words).strip(), tokens=[t for t, _ in words], timestamps=[round(a - origin, 6) for _, a in words])
def filler(name, start, stop, step=0.5):
    return [(f" {name}{i}", start + step * i) for i in range(int(round((stop - start) / step)) + 1)]
class Redo:
    def __init__(self, answers): self.answers, self.asked = list(answers), []
    async def submit_many(self, pieces, key):
        self.asked += [(p[0] / 16000, (p[0] + p.size) / 16000) for p in pieces]
        return [self.answers.pop(0) if self.answers else heard(0, []) for _ in pieces]
for early in (0.32, 0.4, 0.48, 0.56):
    before = filler("a", 0.5, 19.5)
    after = [(" Rome", 28.0), (" fell.", 28.24), (" Then", 29.2), (" Caesar", 29.6)] + filler("b", 30.4, 39.5)
    prep = lib.prepared(40.0, [(0, 40)], [(0, 40)], None)
    routes.speech_segments = lambda wav: [(0, wav.size)]
    redo = heard(17.82, [(" a38", 19.5), (" the", 20.4), (" past", 24.0), (" Rome", 28.0), (" fell.", 28.24), (" Then", 29.2 - early), (" Caesar", 29.6 - early)])
    w = Redo([redo])
    res = asyncio.run(routes._redo_stalled(lib.request(w), [prep], [heard(0.0, before + after)], "parakeet-v3:fp32"))
    text, segs, words = routes._stitch(prep, res)
    print(f"early {early}: redo {[(round(a,2), round(b,2)) for a, b in w.asked]}", [(x["word"], round(x["start"], 2)) for x in words if 27.5 <= x["start"] <= 30.5])
