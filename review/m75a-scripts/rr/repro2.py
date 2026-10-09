import sys
sys.argv = sys.argv[:1]
exec(open(sys.path[0] + "/repro.py").read().split("cases = {")[0])
cases = {
 "2 breaths 300ms -42 @42.5,@47 (10s)": (10, [(42.5, 0.3, -42, "n"), (47.0, 0.3, -42, "n")]),
 "2 breaths 300ms -42 @43,@46 (10s)": (10, [(43, 0.3, -42, "n"), (46, 0.3, -42, "n")]),
 "3 breaths 250ms -42 @43,@48,@53 (15s)": (15, [(43, 0.25, -42, "n"), (48, 0.25, -42, "n"), (53, 0.25, -42, "n")]),
 "breaths 300ms -45 every 4s (30s pause)": (30, [(42 + 4 * k, 0.3, -45, "n") for k in range(7)]),
 "breaths 200ms -45 every 4s (30s pause)": (30, [(42 + 4 * k, 0.2, -45, "n") for k in range(7)]),
}
for name, (pause, ev) in cases.items():
    p = chunker.plan_chunks(make(pause, ev), target_sec=60, max_sec=75, context_sec=5.0)
    r = [(round(a/SR, 2), round(b/SR, 2)) for a, b in p.ranges]
    w = [(round(a/SR, 2), round(b/SR, 2)) for a, b in p.windows]
    print(f"{name:40s} ranges={r}\n{'':40s} windows={w}")
