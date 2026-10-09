"""Interview turn: a quieter talker (off-mic) speaks just before (lead) or just after (tail) a long unbroken answer,
with room tone at the file's edges. Real volume VAD. Quiet talker's words inside a range: main vs head.
usage: margin_turn.py GEN N [CFG]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from common import MAIN, HEAD, SR, plan
from gens import gen_a, gen_b

G = gen_a if sys.argv[1] == "a" else gen_b
N = int(sys.argv[2]); cfg = sys.argv[3] if len(sys.argv) > 3 else "v2 ctx5"
ex = 0
for side in os.environ.get("SIDES", "lead,tail").split(","):
    for lq in [int(x) for x in os.environ.get("LQS", "-30,-34,-38,-42").split(",")]:
        for gdb in (6, 10, 15):
            T = Counter()
            for seed in range(N):
                r = np.random.default_rng([seed, int(-lq), gdb, 1 if side == "lead" else 2, ord(sys.argv[1])])
                edge = r.uniform(0.1, 1.0); Qd = r.uniform(1.5, 5.0); turn = r.uniform(0.05, 0.3); L = r.uniform(36, 39.5)
                if side == "lead":
                    qsp = (edge, edge + Qd); sp = (qsp[1] + turn, qsp[1] + turn + L); total_s = sp[1] + r.uniform(0.3, 3)
                else:
                    sp = (edge, edge + L); qsp = (sp[1] + turn, sp[1] + turn + Qd); total_s = qsp[1] + r.uniform(0.1, 1.0)
                total = int(total_s * SR)
                S = lambda x: int(x * SR)
                if G is gen_a:
                    wav, _ = G([(S(sp[0]), S(sp[1]))], total, r)
                    q, qw = G([(S(qsp[0]), S(qsp[1]))], total, r, gap_db=gdb, base_db=lq, room_db=-120)
                else:
                    wav, _ = G([(S(sp[0]), S(sp[1]))], total, r)
                    q, qw = G([(S(qsp[0]), S(qsp[1]))], total, r, base_db=lq, room_db=-140, gap_db=(gdb - 1, gdb + 1))
                a0, b0 = S(qsp[0]), S(qsp[1])
                wav[a0:b0] += q[a0:b0]
                res = {}
                for nm, mod in (("m", MAIN), ("h", HEAD)):
                    p = plan(mod, wav, cfg, gate=None)
                    cov = np.zeros(total, bool)
                    for a, b in p.ranges: cov[a:b] = True
                    res[nm] = [bool(cov[a:b].all()) for a, b in qw]
                    res[nm + "p"] = p
                heard = sum(any(s < b and e > a for s, e in res["mp"].speech) for a, b in qw)
                T["words"] += len(qw); T["heard"] += heard; T["m"] += sum(res["m"]); T["h"] += sum(res["h"])
                lost = sum(x and not y for x, y in zip(res["m"], res["h"]))
                T["files_fewer"] += lost > 0; T["lost"] += lost; T["split"] += len(res["mp"].ranges) > 1
                if lost and ex < 2 and lq == -34:
                    ex += 1
                    print(f"  EX {side} seed {seed} quiet {qsp[0]:.2f}-{qsp[1]:.2f} @{lq} gaps {gdb}: VAD {[(round(s/SR,2), round(e/SR,2)) for s,e in res['mp'].speech]} main {[(round(a/SR,2), round(b/SR,2)) for a,b in res['mp'].ranges]} head {[(round(a/SR,2), round(b/SR,2)) for a,b in res['hp'].ranges]} lost words {[(round(a/SR,2), round(b/SR,2)) for (a,b),x,y in zip(qw,res['m'],res['h']) if x and not y]}", flush=True)
            print(f"gen {sys.argv[1]} {cfg} {side} quiet talker {lq} dBFS gaps {gdb} dB: files {N} (split {T['split']}), words {T['words']} VAD heard {T['heard']}, "
                  f"in a range main {T['m']} head {T['h']}; head fewer in {T['files_fewer']} files ({T['lost']} words)", flush=True)
