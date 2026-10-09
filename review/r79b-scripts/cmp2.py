import json, sys
from collections import Counter
SUF = "~^*+=<>|"
def ident(word):
    base = word.rstrip(SUF); suf = word[len(base):]
    if not suf: return None
    n = 0
    for ch in suf: n = n * 8 + SUF.index(ch)
    return n
A = {j["case"]: j for j in map(json.loads, open(sys.argv[1]))}
B = {j["case"]: j for j in map(json.loads, open(sys.argv[2]))}
tot = Counter(); bad = []
for c in sorted(A):
    x, y = A[c], B[c]
    if "crash" in x or "crash" in y:
        tot["crash_main" if "crash" in x else "crash_head"] += 1
        bad.append((c, "CRASH", x.get("crash"), y.get("crash"), y.get("tb")))
        continue
    mi = [ident(w[0]) for w in x["words"]]; hi = [ident(w[0]) for w in y["words"]]
    cm, ch = Counter(i for i in mi if i is not None), Counter(i for i in hi if i is not None)
    lost = [i for i in cm if ch[i] < cm[i]]
    dup = [i for i in ch if ch[i] > max(1, cm[i])]
    inv = lambda ids: sum(1 for a, b in zip(ids, ids[1:]) if b < a)
    hids = [i for i in hi if i is not None]; mids = [i for i in mi if i is not None]
    new_inv = inv(hids) - inv(mids)
    inv_m = sum(1 for i in mi if i is None); inv_h = sum(1 for i in hi if i is None)
    tot["main_words"] += len(set(mids)); tot["head_words"] += len(set(hids))
    tot["main_dupwords"] += sum(1 for i in cm if cm[i] > 1); tot["head_dupwords"] += sum(1 for i in ch if ch[i] > 1)
    for k in ("unsorted", "mismatch", "seg_mismatch", "bad_span"):
        tot["main_" + k] += int(x[k]); tot["head_" + k] += int(y[k])
    if lost or dup or new_inv > 0 or inv_h > inv_m or y["unsorted"] > x["unsorted"] or y["mismatch"] or y["seg_mismatch"]:
        tot["head_worse"] += 1
        tot["lost"] += len(lost); tot["dup"] += len(dup); tot["inv"] += max(0, new_inv); tot["noid"] += max(0, inv_h - inv_m)
        bad.append((c, y["pieces"], "lost", lost, "dup", dup, "new_inv", new_inv, "noid", inv_h - inv_m, "unsorted", y["unsorted"], "mm", y["mismatch"], y["seg_mismatch"]))
print(dict(tot))
for b in bad[: int(sys.argv[3]) if len(sys.argv) > 3 else 12]:
    print(b)
