from h import *
cases = [([(0, 18.8), (21.5, 43.5)], 43.5), ([(0, 10), (11.5, 51.45), (54, 93.9)], 93.9), ([(2.93, 4.25), (4.58, 60.73)], 60.73)]
for segs, total in cases:
    for k, m in MODS.items():
        r, w, sp, t = run(m, segs, total, CONFIGS["v2c5"])
        x = metrics(r, w, sp, t, CONFIGS["v2c5"])
        print(k, segs, sec(r), {a: x[a] for a in ("n", "isc", "silent", "short", "wedge")})
