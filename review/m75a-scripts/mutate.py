import subprocess, sys, pathlib
W = pathlib.Path(sys.argv[1]); PY = sys.argv[2]
src_path = W / "parakeet_service/chunker.py"
orig = src_path.read_text()
muts = {
    "median->mean": [("heard = np.median(np.lib.stride_tricks.sliding_window_view(padded, _SYLLABLE), axis=1) > gate",
                      "heard = np.mean(np.lib.stride_tricks.sliding_window_view(padded, _SYLLABLE), axis=1) > gate")],
    "HEARD_ENOUGH=15": [("_HEARD_ENOUGH = 25", "_HEARD_ENOUGH = 15")],
    "HEARD_ENOUGH=20": [("_HEARD_ENOUGH = 25", "_HEARD_ENOUGH = 20")],
    "HEARD_ENOUGH=22": [("_HEARD_ENOUGH = 25", "_HEARD_ENOUGH = 22")],
    "HEARD_ENOUGH=28": [("_HEARD_ENOUGH = 25", "_HEARD_ENOUGH = 28")],
    "HEARD_ENOUGH=30": [("_HEARD_ENOUGH = 25", "_HEARD_ENOUGH = 30")],
    "HEARD_ENOUGH=35": [("_HEARD_ENOUGH = 25", "_HEARD_ENOUGH = 35")],
    "summed (e8811de rule)": [("        kept = heard & near\n", "        kept = heard if heard.sum() >= _HEARD_ENOUGH else np.zeros_like(heard)\n")],
    "no near (sound itself only)": [("near[max(0, a - relisten): b + relisten] = True", "near[a:b] = True")],
    "near = relisten//3": [("near[max(0, a - relisten): b + relisten] = True", "near[max(0, a - relisten // 3): b + relisten // 3] = True")],
    "no joining in loud_frames": [("for a, b in _joined(runs(heard), _dip_frames()).tolist():", "for a, b in runs(heard).tolist():")],
    "join dip x3": [("for a, b in _joined(runs(heard), _dip_frames()).tolist():", "for a, b in _joined(runs(heard), 3 * _dip_frames()).tolist():")],
    "max sound len instead of heard sum": [("if heard[a:b].sum() >= _HEARD_ENOUGH:", "if b - a >= _HEARD_ENOUGH:")],
    "SYLLABLE=3": [("_SYLLABLE = 5  # frames", "_SYLLABLE = 3  # frames")],
    "no median (SYLLABLE=1)": [("_SYLLABLE = 5  # frames", "_SYLLABLE = 1  # frames")],
}
try:
    for name, reps in muts.items():
        s = orig
        for a, b in reps:
            assert s.count(a) == 1, (name, a)
            s = s.replace(a, b)
        src_path.write_text(s)
        r = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_chunker.py", "tests/test_retime.py", "-rf"],
                           cwd=W, capture_output=True, text=True)
        fails = [l.split(" - ")[0].replace("FAILED ", "") for l in r.stdout.splitlines() if l.startswith("FAILED")]
        print(f"{name}: {len(fails)} failing" + "".join(f"\n    {f}" for f in fails))
finally:
    src_path.write_text(orig)
