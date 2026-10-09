import sys
sys.argv = [sys.argv[0], sys.argv[1], "0", sys.argv[2], sys.argv[3]] + sys.argv[5:]
import diff as D
from lib import *
seed, it, mode, ms = int(sys.argv[1]), int(__import__("os").environ["IT"]), sys.argv[3], int(sys.argv[4])
D.rng = np.random.default_rng(seed); D.mode = mode; D.seed = seed
for n in MODS: MODS[n][0].VAD_MIN_SILENCE_MS = ms
for i in range(it + 1):
    wav, words, P = D.gen(i)
print("pause 40 -", round(40 + P, 2), "words", [(round(a, 2), round(b, 2)) for a, b in words])
for n in ("main", "old", "head"):
    for ctx in (0.0, 5.0):
        r = plan(n, wav, ctx).ranges
        print(f"{n} ctx{ctx}: ranges {[x for x in secs(r) if x[1] > 39 and x[0] < 41 + P]} score {D.score_plan(r, words)}")
    ps = [(round(a, 2), round(b, 2)) for a, b in pauses(n, wav) if b > 40 and a < 40 + P]
    print(f"{n} pauses {ps} bad {D.score_pauses(pauses(n, wav), words)}")
