# A quiet speaker's halting turn between two loud turns, at several
# PARAKEET_VAD_MIN_SILENCE_MS: how many of its words are decoded?
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/k78a-scripts")
from lib import *
import diff as D
def case(name, words_db, gap, wlen, n, floor=-55, P=12.0, seed=0):
    t = 41.0; sounds = []
    for i in range(n):
        sounds.append((t, wlen, words_db)); t += wlen + gap
    wav = build(P, sounds, seed=seed, floor_db=floor)
    words = [(s, s + d) for s, d, l in sounds]
    return wav, words
scen = [
    ("0.3 s words, 0.6 s apart, -40 dBFS (15 dB over floor)", -40, 0.6, 0.3, 10),
    ("0.35 s words, 0.5 s apart, -42 dBFS", -42, 0.5, 0.35, 10),
    ("0.25 s words, 0.8 s apart, -38 dBFS", -38, 0.8, 0.25, 8),
    ("0.4 s words, 0.45 s apart, -40 dBFS", -40, 0.45, 0.4, 10),
]
for label, lvl, gap, wl, n in scen:
    wav, words = case(label, lvl, gap, wl, n)
    row = []
    for ms in (400, 600, 800, 1000, 2000):
        for nm in ("main", "head"):
            MODS[nm][0].VAD_MIN_SILENCE_MS = ms
        km = D.score_plan(plan("main", wav, 5.0).ranges, words)[0]
        kh = D.score_plan(plan("head", wav, 5.0).ranges, words)[0]
        row.append(f"{ms}ms main {km}/{n} head {kh}/{n}")
    print(label, "|", "; ".join(row))
