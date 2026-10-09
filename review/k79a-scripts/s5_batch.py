"""Batch: several files, the redo's words land in the right file; two stretches in one piece."""
import asyncio
import sys

sys.path.insert(0, sys.argv[1])
sys.path.insert(1, sys.argv[2])
import numpy as np  # noqa: E402
import simlib  # noqa: E402
from simlib import Model, S, words_every, word_list  # noqa: E402
from parakeet_service import routes  # noqa: E402

routes.speech_segments = lambda wav: [(0, wav.size)]


class Multi:
    """Each file its own model, told apart by the piece's array base."""

    def __init__(self, models):
        self.models = models

    async def submit_many(self, pieces, _key):
        out = []
        for piece in pieces:
            base = piece.base if piece.base is not None else piece
            for wav, model in self.models:
                if base is wav or np.shares_memory(piece, wav):
                    a = float(piece[0]) / 16000
                    out.append(model.decode(a, a + piece.size / 16000))
                    break
        return out


files, models, truths = [], [], []
for k in range(3):
    total = 50.0
    p = simlib.prepared(total, [(0, total)], [(0, total)], None)
    truth = words_every(0.3, 50, 0.6, prefix=f"f{k}_")
    if k == 1:
        skips = [(3.0, 9.0), (20.0, 30.0)]
        m = Model(truth, skips=lambda a, b, s=skips: s if (a, b) == (0, 50) else [])
    else:
        m = Model(truth)
    files.append(p)
    models.append((p.waveform, m))
    truths.append(truth)

worker = Multi(models)


async def go():
    flat = await worker.submit_many([piece for f in files for piece in f.pieces], "x")
    return await routes._redo_stalled(simlib.request(worker), files, flat, "parakeet-v3:fp32")


results = asyncio.run(go())
for k, (f, r, truth) in enumerate(zip(files, results, truths)):
    text, _segs, words = routes._stitch(f, [r])
    want = [w for w, _t in word_list(truth)]
    got = [w["word"] for w in words]
    missing = [w for w in want if w not in got]
    foreign = [w for w in got if not w.startswith(f"f{k}_")]
    starts = [w["start"] for w in words]
    print(f"file {k}: {len(got)}/{len(want)} missing={missing} foreign={foreign} "
          f"ordered={starts == sorted(starts)} calls={models[k][1].calls}")
