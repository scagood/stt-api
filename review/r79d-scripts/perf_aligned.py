import time, random
from types import SimpleNamespace
import lib
from lib import routes
SR = 16000
def res(words, origin=0.0):
    return SimpleNamespace(text="", tokens=[w for w, _ in words], timestamps=[round(t - origin, 2) for _, t in words])
for label, step, vocab in (("distinct words, 0.3 s apart", 0.3, None), ("all 'la', 0.3 s apart", 0.3, ["la"]), ("all 'la', 0.16 s apart", 0.16, ["la"]), ("'no'/'the' mix, 0.2 s", 0.2, ["no", "the"])):
    total = 75.0
    n = int(total / step)
    rng = random.Random(1)
    words = [(" " + (rng.choice(vocab) if vocab else f"w{i}"), round(0.2 + i * step, 2)) for i in range(n) if 0.2 + i * step < total - 0.5]
    # piece heard every other 3.5 s block: stretches throughout -> one merged window ~ whole clip
    piece = [w for w in words if int(w[1] / 3.5) % 2 == 0]
    prep = lib.prepared(total, [(0, total)], [(0, total)], None)
    routes.speech_segments = lambda wav: [(0, wav.size)]
    stretches = routes._stalled(prep, [res(piece)])[0]
    wins = routes._redo_windows(prep.windows[0], stretches)
    t0 = time.perf_counter()
    merged = res(piece)
    for a, b in wins:
        redo = res([w for w in words if a / SR <= w[1] < b / SR], a / SR)
        inside = [(lo, hi) for lo, hi in stretches if a <= lo and hi <= b]
        merged, added = routes._merged(merged, 0, redo, a, b, inside, prep.ranges[0], [])
    dt = time.perf_counter() - t0
    print(f"{label}: {len(words)} words, piece {len(piece)}, {len(stretches)} stretches, windows {[(round(a/SR,1), round(b/SR,1)) for a,b in wins]}, splice {dt*1000:.0f} ms, words after {len(routes._word_spans(merged.tokens))}")
