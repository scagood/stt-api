"""Compare cutfz.py outputs: python cutcmp.py main.jsonl head.jsonl [show]"""
import json
import sys
from collections import Counter

DIG = "#$%&*+=@"


def untag(word):
    t = word.rstrip("~")
    tail = t[-4:]
    if len(t) < 5 or any(c not in DIG for c in tail):
        return None
    v = 0
    for c in tail:
        v = v * 8 + DIG.index(c)
    return v


def load(path):
    return {r["case"]: r for r in map(json.loads, open(path))}


def copies(words):
    c = Counter()
    for k, (w, _s) in enumerate(words):
        i = untag(w)
        if i is None:
            continue
        if w.endswith("~") and k and untag(words[k - 1][0]) == i and not words[k - 1][0].endswith("~"):
            continue  # second half of a split reading, after its first half
        c[i] += 1
    return c


A, B = load(sys.argv[1]), load(sys.argv[2])
show = int(sys.argv[3]) if len(sys.argv) > 3 else 6
T = Counter()
EX = {}


def note(kind, info, n=1):
    T[kind] += n
    EX.setdefault(kind, []).append(info)


for case in sorted(A):
    a, b = A[case], B.get(case)
    if b is None:
        continue
    if "crash" in a:
        note("main_crash", (case, a["crash"]))
        continue
    if "crash" in b:
        note("HEAD_CRASH", (case, b["crash"], b.get("tb")))
        continue
    T["cases"] += 1
    T["main_decodes"] += len(a["calls"])
    T["head_decodes"] += len(b["calls"])
    for fid, (fa, fb) in enumerate(zip(a["files"], b["files"])):
        truth = {i: (x, t) for i, x, t in fa["truth"]}
        T["truth_words"] += len(truth)
        for p in fb["problems"]:
            note("HEAD_" + p, (case, fid))
        for p in fa["problems"]:
            T["main_" + p] += 1
        wa, wb = [tuple(x) for x in fa["words"]], [tuple(x) for x in fb["words"]]
        ca, cb = copies(wa), copies(wb)
        lost = sorted(set(ca) - set(cb))
        if lost:
            note("LOST_vs_main", (case, fid, a["kinds"][fid], [(i, truth[i]) for i in lost[:6]]), len(lost))
        dup = sorted(i for i in cb if i in ca and cb[i] > max(1, ca[i]))
        if dup:
            note("DUP_of_main_word", (case, fid, a["kinds"][fid], [(i, truth[i], [w for w in wb if untag(w[0]) == i]) for i in dup[:6]]), len(dup))
        dupnew = sorted(i for i in cb if i not in ca and cb[i] > 1)
        if dupnew:
            note("dup_of_new_word", (case, fid, a["kinds"][fid], [(i, truth[i], [w for w in wb if untag(w[0]) == i]) for i in dupnew[:6]]), len(dupnew))
        # order of main's words in head
        pos = {}
        for k, (w, _s) in enumerate(wb):
            i = untag(w)
            if i is not None and i not in pos:
                pos[i] = k
        seq, seen = [], set()
        for w, _s in wa:
            i = untag(w)
            if i is not None and i in pos and i not in seen:
                seen.add(i)
                seq.append(pos[i])
        inv = sum(1 for x, y in zip(seq, seq[1:]) if y < x)
        if inv:
            note("REORDER_of_main_words", (case, fid, inv), inv)
        # truth-order inversions
        def inversions(ws):
            ids = [untag(w) for w, _s in ws if untag(w) is not None]
            return sum(1 for x, y in zip(ids, ids[1:]) if y < x)
        ia, ib = inversions(wa), inversions(wb)
        T["main_truth_inversions"] += ia
        T["head_truth_inversions"] += ib
        if ib > ia:
            note("head_more_truth_inversions", (case, fid, ia, ib))
        T["main_missing"] += len(set(truth) - set(ca))
        T["head_missing"] += len(set(truth) - set(cb))
        inv_a = [w for w in wa if untag(w[0]) is None]
        inv_b = [w for w in wb if untag(w[0]) is None]
        T["main_invented"] += len(inv_a)
        T["head_invented"] += len(inv_b)
        if len(inv_b) > len(inv_a):
            note("HEAD_MORE_INVENTED", (case, fid, a["kinds"][fid], [w for w in inv_b if w not in inv_a][:4]), len(inv_b) - len(inv_a))
        ea = sum(c - 1 for c in ca.values() if c > 1)
        eb = sum(c - 1 for c in cb.values() if c > 1)
        T["main_extra_copies"] += ea
        T["head_extra_copies"] += eb
        for w, s in wb:
            i = untag(w)
            if i is not None and abs(s - truth[i][1]) > 0.8:
                note("head_ts_off_0.8", (case, fid, w, s, truth[i][1]))
        for w, s in wa:
            i = untag(w)
            if i is not None and abs(s - truth[i][1]) > 0.8:
                T["main_ts_off_0.8"] += 1
        T["files"] += 1
        T["files_same"] += wa == wb

for k in sorted(T):
    print(f"{k}: {T[k]}")
for k, ex in EX.items():
    if k.isupper() or k.startswith(("LOST", "DUP", "REORDER", "HEAD", "dup_of_new", "head_")):
        print(f"--- {k} ({len(ex)})")
        for e in ex[:show]:
            print("   ", str(e)[:700])
