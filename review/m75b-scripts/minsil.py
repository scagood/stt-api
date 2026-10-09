import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/m75b-scripts")
from lib import *
from quietturn import scenario, covered
for ms in (20, 60, 100, 200, 400, 1000, 2500):
    for n in ("prev", "head"):
        MODS[n][0].VAD_MIN_SILENCE_MS = ms
    out = []
    for level, floor in ((-40, -70), (-40, -60), (-44, -55)):
        res = {"prev": [0, 0], "head": [0, 0]}; tot = 0
        for seed in range(3):
            wav, a, b, words = scenario(level, floor, seed, False)
            tot += len(words)
            for n in res:
                ch, rt = MODS[n]
                rr = ch.plan_chunks(wav, **BOUNDS, context_sec=5.0).ranges
                res[n][0] += sum(covered(rr, x, y) == y - x for x, y in words)
                ps = [(int(p * SR), int(q * SR)) for p, q in rt.pauses(wav)]
                res[n][1] += sum(any(p <= x and y <= q for p, q in ps) for x, y in words)
        out.append(f"{level}/{floor}: {tot} dec p/h {res['prev'][0]}/{res['head'][0]} inpause p/h {res['prev'][1]}/{res['head'][1]}")
    # breaths every 2.5 s, 300 ms at -42 in 20 s pause
    wav = build(20, [(41.0 + 2.5 * k, 0.3, -42) for k in range(1, 8)])
    out.append("breaths: prev " + str(plan("prev", wav)) + " head " + str(plan("head", wav)))
    print(f"VAD_MIN_SILENCE_MS={ms}:\n   " + "\n   ".join(out))
