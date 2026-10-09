"""Compare by runs: consecutive truth words with one base are one run, counted."""
import json, sys
from collections import Counter
SUF = "~^*+=<>|"
def ident(word):
    base = word.rstrip(SUF); suf = word[len(base):]
    if not suf: return None
    n = 0
    for ch in suf: n = n * 8 + SUF.index(ch)
    return n
def runs(truth):
    rid, out = -1, []
    for k, b in enumerate(truth):
        if k == 0 or b != truth[k - 1]: rid += 1
        out.append(rid)
    return out
A = {j["case"]: j for j in map(json.loads, open(sys.argv[1]))}
B = {j["case"]: j for j in map(json.loads, open(sys.argv[2]))}
tot = Counter(); bad = []
for c in sorted(A):
    x, y = A[c], B[c]
    if "crash" in x or "crash" in y:
        tot["crash_main" if "crash" in x else "crash_head"] += 1
        bad.append((c, "CRASH", x.get("crash"), y.get("crash"), y.get("tb")))
        continue
    R = runs(x["truth"])
    # a word's run; alt "x"/"y" words keep their id; same-id variants (split fragments) counted separately
    def key(w):
        i = ident(w)
        if i is None: return None
        base = w.rstrip(SUF)
        variant = "y" if base.startswith("y") and not x["truth"][i].startswith("y") else ""
        return (R[i], variant)
    mk = [key(w[0]) for w in x["words"]]; hk = [key(w[0]) for w in y["words"]]
    cm, ch = Counter(k for k in mk if k), Counter(k for k in hk if k)
    lost = [k for k in cm if ch[k] < cm[k]]
    # a run (any variant) heard more often than in truth
    truth_n = Counter(R)
    over = lambda cnt, k: sum(v for (r, var), v in cnt.items() if r == k[0])
    dup = [k for k in ch if ch[k] > cm[k] and (over(ch, k) > truth_n[k[0]])]
    inv = lambda ks: sum(1 for a, b in zip(ks, ks[1:]) if a[0] > b[0])
    hs = [k for k in hk if k]; ms = [k for k in mk if k]
    new_inv = inv(hs) - inv(ms)
    noid = sum(1 for k in hk if k is None) - sum(1 for k in mk if k is None)
    tot["cases"] += 1
    tot["main_runs_heard"] += sum(cm.values()); tot["head_runs_heard"] += sum(ch.values())
    for k in ("unsorted", "mismatch", "seg_mismatch", "bad_span"):
        tot["head_" + k] += int(y[k]); tot["main_" + k] += int(x[k])
    if lost or dup or new_inv > 0 or noid > 0 or y["unsorted"] > x["unsorted"] or y["mismatch"] or y["seg_mismatch"]:
        tot["head_worse_cases"] += 1
        tot["lost"] += sum(cm[k] - ch[k] for k in lost); tot["dup"] += len(dup); tot["inv"] += max(0, new_inv); tot["noid"] += max(0, noid)
        bad.append((c, y["pieces"], "lost", lost, "dup", dup, "new_inv", new_inv, "noid", noid))
print(dict(tot))
for b in bad[: int(sys.argv[3]) if len(sys.argv) > 3 else 12]:
    print(b)
