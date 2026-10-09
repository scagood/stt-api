"""Rerun one cutfz case with tracing: python cutdbg.py <wt> <mode> <seed> <case> <lo> <hi> [k=v...]"""
import sys

wt, mode, seed, case, lo, hi = sys.argv[1:7]
extra = sys.argv[7:]
sys.argv = [sys.argv[0], wt, mode, seed, "0", "/dev/null"] + extra
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("cutfz", __file__.replace("cutdbg.py", "cutfz.py"))
src = open(spec.origin).read().replace("\nmain()\n", "\n")
g = {"__name__": "cutfz"}
exec(compile(src, spec.origin, "exec"), g)
routes, random, asyncio, SR = g["routes"], g["random"], g["asyncio"], g["SR"]
lo, hi, case = float(lo), float(hi), int(case)

calls = []
orig_merged = getattr(routes, "_merged", None)


def show(tag, res, origin):
    info = routes._extract(res)
    ws = [(w, round(origin + info["timestamps"][f], 2)) for w, f, _l in routes._word_spans(info["tokens"])]
    print(f"  {tag}:", [x for x in ws if lo <= x[1] <= hi])


if orig_merged:
    def traced(result, origin, again, start, stop, *a, **k):
        out, added = orig_merged(result, origin, again, start, stop, *a, **k)
        print(f"_merged origin={origin/SR:.2f} redo={start/SR:.2f}-{stop/SR:.2f} args={[(round(x/SR,2), round(y/SR,2)) for x, y in a[0]]} own={[round(x/SR,2) for x in a[1]]} taken={[(round(x/SR,2), round(y/SR,2), z) for x, y, z in (a[2] if len(a) > 2 else k.get('taken', ()))]}")
        show("piece", result, origin / SR)
        show("redo ", again, start / SR)
        show("out  ", out, origin / SR)
        print("  added", [round(x / SR, 2) for x in added])
        return out, added
    routes._merged = traced

rng = random.Random(f"{seed}-{case}-layout")
kinds = [rng.choice(["one", "two", "three"]) for _ in range(3)] if mode == "batch" else [mode]
files = [g["File"](rng, k, i) for i, k in enumerate(kinds)]
model = g["Model"](case, files)
vads = {id(f.wav): [(int(round(a * SR)), int(round(b * SR))) for a, b in f.vad] for f in files}
if hasattr(routes, "speech_segments"):
    routes.speech_segments = lambda wav: vads[id(wav)]
first = []
for f in files:
    print("file", f.fid, f.kind, "ranges", f.ranges, "windows", f.windows, "first skips", [[(round(a, 2), round(b, 2)) for a, b in s] for s in f.first_skips], "spots", [(round(a, 2), round(b, 2), round(L, 1)) for a, b, L in f.spots])
    print("  truth:", [(x, round(t, 2)) for i, x, _t, times in f.words for t in [times[0]] if lo <= t <= hi])
    for (a, b), sk in zip(f.windows, f.first_skips):
        r = model.decode(f, a, b, first_skips=sk)
        first.append(r)
        show(f"first {a}-{b}", r, a)
results = asyncio.run(routes._redo_stalled(g["request"](g["Worker"](model)), [f.prep for f in files], first, "parakeet-v3:fp32"))
print("calls", model.calls)
cursor = 0
for f in files:
    n = len(f.prep.pieces)
    for k in range(n):
        show(f"final piece {k} (window {f.prep.windows[k][0]/SR:.2f}-{f.prep.windows[k][1]/SR:.2f})", results[cursor + k], f.prep.windows[k][0] / SR)
    kept = routes._trimmed(f.prep, results[cursor:cursor + n])
    for k in range(n):
        show(f"trimmed {k}", kept[k], f.prep.ranges[k][0] / SR)
    text, segs, words = routes._stitch(f.prep, results[cursor:cursor + n])
    print("  OUT:", [(w["word"], round(w["start"], 2)) for w in words if lo <= w["start"] <= hi])
    cursor += n
