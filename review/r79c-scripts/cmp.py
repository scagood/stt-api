"""Compare fz.py outputs: python cmp.py main.jsonl head.jsonl [show]"""
import json
import re
import sys
from collections import Counter

IDX = re.compile(r"^[wWm](\d+)")
SPLIT_B = re.compile(r"^m\d+b$")


def idx_of(word):
    m = IDX.match(word)
    return int(m.group(1)) if m else None


def load(path):
    return {r["case"]: r for r in map(json.loads, open(path))}


A, B = load(sys.argv[1]), load(sys.argv[2])
show = int(sys.argv[3]) if len(sys.argv) > 3 else 8
T = Counter()
examples = {}


def note(kind, info):
    T[kind] += 1
    examples.setdefault(kind, []).append(info)


for case in sorted(A):
    a, b = A[case], B.get(case)
    if b is None:
        continue
    if "crash" in a:
        note("main_crash", (case, a["crash"]))
        continue
    if "crash" in b:
        note("head_crash", (case, b["crash"], b.get("tb")))
        continue
    T["cases"] += 1
    ca = [c for c in a["calls"]]
    cb = [c for c in b["calls"]]
    T["main_decodes"] += len(ca)
    T["head_decodes"] += len(cb)
    T["main_sec"] += sum(c[2] - c[1] for c in ca)
    T["head_sec"] += sum(c[2] - c[1] for c in cb)
    T["audio_sec"] += sum(t[-1][2] if t else 0 for t in [])
    T["head_vad_calls"] += b.get("vad_calls", 0)
    if cb != ca[: len(cb)] or len(cb) > len(ca):
        T["cases_more_decodes"] += len(cb) > len(ca)
    for fa, fb in zip(a["files"], b["files"]):
        truth = {i: t for i, _w, t in fa["truth"]}
        T["truth_words"] += len(truth)
        T["audio_sec"] += fa["windows"][-1][1]
        for p in fb["problems"]:
            note("head_" + p, (case, fb["text"][:80]))
        for p in fa["problems"]:
            T["main_" + p] += 1
        for name, po in (fb.get("paths") or {}).items():
            T["paths_run"] += 1
            if "crash" in po:
                note(f"head_path_{name}_crash", (case, po["crash"]))
            for p in po.get("problems", []):
                note(f"head_path_{name}_{p}", case)
        for name, po in (fa.get("paths") or {}).items():
            if "crash" in po:
                T[f"main_path_{name}_crash"] += 1
            for p in po.get("problems", []):
                T[f"main_path_{name}_{p}"] += 1
        wa = [(w, s) for w, s, _e in fa["words"]]
        wb = [(w, s) for w, s, _e in fb["words"]]
        ia = [idx_of(w) for w, _s in wa]
        ib = [idx_of(w) for w, _s in wb]
        # copies of each word: the second half of a split mishear ("m12a m12b", one decode) is not a copy
        def copies(ws):
            """Copies of each word: a split mishear (m12a m12b, one decode) is one; a lone m12b is one."""
            names = [w for w, _s in ws]
            c = Counter()
            for k, w in enumerate(names):
                i = idx_of(w)
                if i is None:
                    continue
                if SPLIT_B.match(w) and k and names[k - 1] == f"m{i}a":
                    continue
                c[i] += 1
            return c
        cnta, cntb = copies(wa), copies(wb)
        # main-kept words that head loses / duplicates / reorders
        lost = sorted(set(cnta) - set(cntb))
        if lost:
            note("LOST_vs_main", (case, fa["kinds"] if "kinds" in fa else a["kinds"], lost[:10], [truth[i] for i in lost[:10]]))
            T["lost_words"] += len(lost)
        dup = sorted(i for i in cntb if cntb[i] > max(1, cnta.get(i, 0)) and i in cnta)
        if dup:
            note("DUP_of_main_word", (case, dup[:10], [truth[i] for i in dup[:10]]))
            T["dup_words"] += len(dup)
        dup_new = sorted(i for i in cntb if cntb[i] > 1 and i not in cnta and cntb[i] > 1)
        # a word main missed, which head has twice
        if dup_new:
            # a split mishear in one decode gives two words: count only where head has 2 distinct copies in addition
            T["dup_new_words"] += len(dup_new)
            examples.setdefault("dup_new_words", []).append((case, dup_new[:6], [truth[i] for i in dup_new[:6]],
                                                             [(w, s) for w, s in wb if idx_of(w) in dup_new[:6]]))
        # order: positions in head of main's words (first occurrence)
        pos = {}
        for k, i in enumerate(ib):
            if i is not None and i not in pos:
                pos[i] = k
        seq = []
        seen = set()
        for i in ia:
            if i is not None and i in pos and i not in seen:
                seen.add(i)
                seq.append(pos[i])
        inv = sum(1 for x, y in zip(seq, seq[1:]) if y < x)
        if inv:
            note("REORDER_of_main_words", (case, inv))
        # inversions vs truth order
        def inversions(ids):
            ids = [i for i in ids if i is not None]
            return sum(1 for x, y in zip(ids, ids[1:]) if y < x)
        T["main_truth_inversions"] += inversions(ia)
        T["head_truth_inversions"] += inversions(ib)
        if inversions(ib) > inversions(ia):
            note("head_more_truth_inversions", (case, inversions(ia), inversions(ib)))
        # missing vs truth
        T["main_missing"] += len(set(truth) - set(cnta))
        T["head_missing"] += len(set(truth) - set(cntb))
        # invented: no index
        inv_a = [w for w, _s in wa if idx_of(w) is None]
        inv_b = [w for w, _s in wb if idx_of(w) is None]
        T["main_invented"] += len(inv_a)
        T["head_invented"] += len(inv_b)
        if len(inv_b) > len(inv_a):
            note("head_more_invented", (case, inv_a, inv_b, [(w, s) for w, s in wb if idx_of(w) is None]))
        # same speech twice: an index with words in head beyond main's count (incl. split mishears)
        ea = sum(c - 1 for c in cnta.values() if c > 1)
        eb = sum(c - 1 for c in cntb.values() if c > 1)
        T["main_extra_copies"] += ea
        T["head_extra_copies"] += eb
        if eb > ea:
            T["HEAD_EXTRA_COPIES_vs_main"] += eb - ea
            note("head_extra_copy_case", (case, [(w, s) for w, s in wb if cntb.get(idx_of(w) or -1, 0) > max(1, cnta.get(idx_of(w) or -1, 0))][:6]))
        # word variant changed for a word main kept
        va = {}
        for w, _s in wa:
            i = idx_of(w)
            if i is not None:
                va.setdefault(i, []).append(w)
        vb = {}
        for w, _s in wb:
            i = idx_of(w)
            if i is not None:
                vb.setdefault(i, []).append(w)
        changed = [i for i in va if i in vb and sorted(va[i]) != sorted(vb[i])]
        if changed:
            note("variant_changed", (case, [(va[i], vb[i]) for i in changed[:5]]))
        # timestamps vs truth
        for w, s in wb:
            i = idx_of(w)
            if i is not None and abs(s - truth[i]) > 0.5:
                note("head_ts_off_0.5", (case, w, s, truth[i]))
        for w, s in wa:
            i = idx_of(w)
            if i is not None and abs(s - truth[i]) > 0.5:
                T["main_ts_off_0.5"] += 1
        # text vs main
        if fa["text"] == fb["text"]:
            T["files_same_text"] += 1
        T["files"] += 1

for k in sorted(T):
    v = T[k]
    print(f"{k}: {round(v, 1) if isinstance(v, float) else v}")
for k, ex in examples.items():
    if k in ("variant_changed",) or k.startswith(("LOST", "DUP", "REORDER", "head_")) or k == "dup_new_words":
        print(f"--- {k} ({len(ex)})")
        for e in ex[:show]:
            print("   ", str(e)[:600])
