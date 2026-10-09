import json
import sys

a, b = sys.argv[1], sys.argv[2]
A = [json.loads(l) for l in open(a)]
B = [json.loads(l) for l in open(b)]
tot = {"main_missing": 0, "pr_missing": 0, "main_dup": 0, "pr_dup": 0, "main_extra": 0, "pr_extra": 0,
       "main_inv": 0, "pr_inv": 0, "pr_worse": 0, "pr_better": 0, "crash": 0, "pr_unsorted": 0, "mismatch": 0}
worse = []
for x, y in zip(A, B):
    if "crash" in x or "crash" in y:
        tot["crash"] += 1
        print("CRASH", x.get("crash"), y.get("crash"), y["case"])
        continue
    tot["main_missing"] += len(x["missing"]); tot["pr_missing"] += len(y["missing"])
    tot["main_dup"] += len(x["dup"]); tot["pr_dup"] += len(y["dup"])
    tot["main_extra"] += len(x["extra"]); tot["pr_extra"] += len(y["extra"])
    tot["main_inv"] += x["inversions"]; tot["pr_inv"] += y["inversions"]
    tot["pr_unsorted"] += y["unsorted"]
    tot["mismatch"] += y["text_words_mismatch"]
    bad_pr = len(y["missing"]) + len(y["dup"]) + len(y["extra"]) + y["inversions"]
    bad_main = len(x["missing"]) + len(x["dup"]) + len(x["extra"]) + x["inversions"]
    if set(y["missing"]) - set(x["missing"]) or set(y["dup"]) - set(x["dup"]) or y["inversions"] > x["inversions"]:
        tot["pr_worse"] += 1
        worse.append((y["case"], y["kind"], y["piece"], y["skips"], "main miss", x["missing"], "pr miss", y["missing"],
                      "main dup", x["dup"], "pr dup", y["dup"], "inv", x["inversions"], y["inversions"], "pr calls", y["calls"], "main calls", x["calls"]))
    elif bad_pr < bad_main:
        tot["pr_better"] += 1
print(tot)
for w in worse[:15]:
    print(w)
