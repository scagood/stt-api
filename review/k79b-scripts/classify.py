import json, sys
A = [json.loads(l) for l in open(sys.argv[1])]
B = [json.loads(l) for l in open(sys.argv[2])]
edge = other = 0
others = []
for x, y in zip(A, B):
    new = [(w, t) for w, t in zip(y["missing"], y["missing_t"]) if w not in x["missing"]]
    for w, t in new:
        # near a skip edge: within 0.45 s of a skip start or end
        if any(abs(t - a) < 0.45 or abs(t - b) < 0.45 for a, b in y["skips"]):
            edge += 1
        else:
            other += 1
            others.append((y["case"], y["kind"], y["piece"], y["skips"], w, t, y["calls"]))
print("new-missing near a skip edge:", edge, "elsewhere:", other)
for o in others[:10]: print(o)
