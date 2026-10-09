"""A redo that raises ModelLoadError (or RuntimeError) for one- and two-piece clips,
on the first redo call or only on the second (the stretch redo after a kept range redo).

python redo_fail2.py <worktree>"""
import asyncio
import logging
import sys

sys.path.insert(0, sys.argv[1])
sys.path.insert(1, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/k79a-scripts")
import simlib  # noqa: E402
from simlib import Model, words_every, word_list, S  # noqa: E402
from parakeet_service import routes  # noqa: E402
from parakeet_service.model import ModelLoadError  # noqa: E402

logging.disable(logging.CRITICAL)


class Failing:
    def __init__(self, model, fail_on, exc):
        self.model, self.n, self.fail_on, self.exc = model, 0, fail_on, exc

    async def submit_many(self, pieces, _key):
        self.n += 1
        if self.n in self.fail_on:
            raise self.exc
        out = []
        for piece in pieces:
            a = float(piece[0]) / 16000
            out.append(self.model.decode(a, a + piece.size / 16000))
        return out


def go(name, total, ranges, windows, speech, model, fail_on, exc):
    if hasattr(routes, "speech_segments"):
        routes.speech_segments = lambda wav: [(0, wav.size)]
    prep = simlib.prepared(total, ranges, windows, speech)
    worker = Failing(model, fail_on, exc)

    async def run():
        first = [model.decode(a / 16000, b / 16000) for a, b in prep.windows]
        return await routes._redo_stalled(simlib.request(worker), [prep], first, "parakeet-v3:fp32")

    try:
        results = asyncio.run(run())
        text, _s, words = routes._stitch(prep, results)
        got = [w["word"] for w in words]
        print(f"{name}: ok, {len(got)} words, redo calls={model.calls[len(prep.pieces):]}, windows={[(a/16000, b/16000) for a, b in prep.windows]}")
        return got
    except Exception as e:  # noqa: BLE001
        print(f"{name}: raised {type(e).__name__} {getattr(e, 'status_code', '')} {e}")


truth = words_every(0.3, 30, 0.6)
for exc in (ModelLoadError("evicted"), RuntimeError("worker died")):
    m = Model(truth, skips=lambda a, b: [(10.0, 20.0)] if (a, b) == (0, 30) else [])
    go(f"one-piece, redo raises {type(exc).__name__}", 30, [(0, 30)], [(0, 30)], None, m, {1}, exc)
    truth2 = words_every(0.3, 120, 0.6)
    m = Model(truth2, skips=lambda a, b: [(20.0, 35.0)] if b - a >= 40 else [])
    go(f"two-piece, range redo raises {type(exc).__name__}", 120, [(0, 60), (60, 120)], [(0, 65), (55, 120)], [(0, 120)], m, {1}, exc)
    m = Model(truth2, skips=lambda a, b: [(20.0, 35.0)] if b - a >= 40 else [])
    go(f"two-piece, stretch redo raises {type(exc).__name__}", 120, [(0, 60), (60, 120)], [(0, 65), (55, 120)], [(0, 120)], m, {2}, exc)
    # range redo kept, then stretch redo (of a second piece... ) raises: the kept range redo must stay
    m = Model(truth2, skips=lambda a, b: ([(20.0, 35.0)] if (a, b) == (0, 65) else []) + ([(80.0, 95.0)] if b - a >= 40 and a > 50 else []))
    got = go(f"two-piece, range redo kept for piece 0, stretch redo for piece 1 raises {type(exc).__name__}", 120,
             [(0, 60), (60, 120)], [(0, 65), (55, 120)], [(0, 120)], m, {2}, exc)
    if got:
        want = [w for w, _t in word_list(truth2)]
        print("   missing:", [w for w in want if w not in got][:5], "... total", len([w for w in want if w not in got]))
