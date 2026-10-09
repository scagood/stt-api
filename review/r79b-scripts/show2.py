import json, sys
SUF = "~^*+=<>|"
def ident(word):
    base = word.rstrip(SUF); suf = word[len(base):]
    if not suf: return None
    n = 0
    for ch in suf: n = n * 8 + SUF.index(ch)
    return n
tag, case, lo, hi = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
for w in ("main", "head"):
    for line in open(f"out/{tag}-{w}.jsonl"):
        j = json.loads(line)
        if j["case"] != case: continue
        if w == "main": print("ranges", j["ranges"])
        print(w, "calls", [c for c in j["calls"] if c[0] != "first"])
        print("  ", [(x[0].rstrip(SUF), x[1], ident(x[0])) for x in j["words"] if lo <= x[1] <= hi])
