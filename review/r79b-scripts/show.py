import json, sys
SUF = "~^*+=<>|"
def ident(word):
    base = word.rstrip(SUF); suf = word[len(base):]
    if not suf: return None
    n = 0
    for ch in suf: n = n * 8 + SUF.index(ch)
    return n
tag, case, ids = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")]
for w in ("main", "head"):
    for line in open(f"out/{tag}-{w}.jsonl"):
        j = json.loads(line)
        if j["case"] != case: continue
        if w == "main": print("ranges", j["ranges"])
        print(w, "calls", j["calls"])
        words = j["words"]
        pos = [k for k, x in enumerate(words) if ident(x[0]) in ids]
        lo = max(0, min(pos or [0]) - 5); hi = max(pos or [0]) + 6
        if not pos:
            # find by neighbour ids
            near = [k for k, x in enumerate(words) if ident(x[0]) is not None and min(ids) - 4 <= ident(x[0]) <= max(ids) + 4]
            lo, hi = (min(near), max(near) + 1) if near else (0, 0)
        print("  ", [(x[0], x[1], ident(x[0])) for x in words[lo:hi]])
