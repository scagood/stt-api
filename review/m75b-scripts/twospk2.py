import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/m75b-scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("t2", "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/m75b-scripts/twospk.py")
src = open(spec.origin).read().split("for style in")[0]
exec(src)
for style in ("halting", "fluent"):
    for b_level, floor in ((-44, -60), (-46, -58), (-44, -55)):
        agg = {n: [0, 0, 0, 0, 0] for n in ("main", "prev", "head")}; totB = totA = 0
        for seed in range(6):
            wav, truth, para = make(seed, b_level, floor, style)
            totB += sum(w["word"].startswith("b") for w in truth); totA += sum(w["word"].startswith("A") for w in truth)
            for n in agg:
                out, ranges = run(n, wav, para)
                dec = [i for i, w in enumerate(para) if any(a / SR <= w["start"] < b / SR for a, b in ranges)]
                errs = [(truth[i]["word"][0], abs(o["start"] - truth[i]["start"])) for i, o in zip(dec, out)]
                agg[n][0] += sum(1 for i, w in enumerate(truth) if w["word"].startswith("b") and i not in dec)
                agg[n][1] += sum(1 for c, e in errs if c == "b" and e > 0.3)
                agg[n][2] += sum(1 for c, e in errs if c == "A" and e > 0.3)
                agg[n][3] += sum(1 for i, w in enumerate(truth) if w["word"].startswith("A") and i not in dec)
        print(f"{style:8s} B{b_level}/floor{floor} (B {totB}, A {totA}): " + "  ".join(f"{n}: Blost {v[0]} Bmoved {v[1]} Alost {v[3]} Amoved {v[2]}" for n, v in agg.items()))
