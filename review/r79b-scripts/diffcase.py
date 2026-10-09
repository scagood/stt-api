import json, sys, difflib
SUF = "~^*+=<>|"
def ident(word):
    base = word.rstrip(SUF); suf = word[len(base):]
    if not suf: return None
    n = 0
    for ch in suf: n = n * 8 + SUF.index(ch)
    return n
tag, case = sys.argv[1], int(sys.argv[2])
only_dup = len(sys.argv) > 3
J = {}
for w in ("main", "head"):
    for line in open(f"out/{tag}-{w}.jsonl"):
        j = json.loads(line)
        if j["case"] == case: J[w] = j
m, h = J["main"], J["head"]
print("ranges", m["ranges"], "head calls", [c for c in h["calls"] if c[0] != "first"])
fmt = lambda x: f"{x[0].rstrip(SUF)}#{ident(x[0])}@{x[1]}"
A = [fmt(x) for x in m["words"]]; B = [fmt(x) for x in h["words"]]
a_ids = [x.split("@")[0] for x in A]; b_ids = [x.split("@")[0] for x in B]
sm = difflib.SequenceMatcher(a=a_ids, b=b_ids, autojunk=False)
for op, i1, i2, j1, j2 in sm.get_opcodes():
    if op == "equal": continue
    if op == "insert":
        ids = [ident(x[0]) for x in h["words"][j1:j2]]
        # only show inserts whose id already exists in head elsewhere (dups) or none id
        if only_dup and all(ids.count(i) == 1 and b_ids.count(b_ids[j1 + k]) == 1 and i is not None and (sum(1 for y in h["words"] if ident(y[0]) == i) == 1) for k, i in enumerate(ids)):
            continue
    print(op, "main:", A[max(0,i1-2):i2+2], "\n      head:", B[max(0,j1-2):j2+2])
