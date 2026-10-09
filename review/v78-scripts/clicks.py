"""Item 4: quiet 1 s phrase at -40 (41-42 s, reach to 45.0), seven 60 ms clicks at -44 every 0.36 s from 44.8 s,
10 s pause at -55 between -20 dBFS turns. Plus a sweep of trains starting just inside the reach."""
import numpy as np
import lib_b as lib
SR = lib.SR
def scene(clicks, P=10.0, seed=4, plain=False):
    rng = np.random.default_rng(seed)
    first = lib.noise(40, -20, rng) if plain else lib.turn(40, -20, rng)
    pause = lib.noise(P, -55, rng)
    lib.add(pause, 1.0, 1.0, -40, rng)
    for at, el, lv in clicks:
        lib.add(pause, at, el, lv, rng)
    last = lib.noise(40, -20, rng) if plain else lib.turn(40, -20, rng)
    return np.concatenate([first, pause, last])
clicks = [(4.8 + 0.36 * k, 0.06, -44) for k in range(7)]
for plain in (False, True):
    for seed in (4, 5, 6):
        wav = scene(clicks, seed=seed, plain=plain)
        print(f"-- seed {seed} {'plain-noise turns' if plain else 'syllabic turns'}")
        for tag in ("main", "old", "head"):
            lr = lib.loud_runs_sec(tag, wav, 42.5, 49.9)
            rg = lib.plan(tag, wav).ranges
            rg60 = lib.plan(tag, wav, target_sec=60.0, max_sec=75.0).ranges
            ps = [(round(a, 2), round(b, 2)) for a, b in lib.pauses(tag, wav) if b > 42 and a < 50]
            print(f"  {tag:4s} split loud past 42.5: {lr}; last {max((b for a,b in lr), default=0):.2f} (= {max((b for a,b in lr), default=45)-45:+.2f} vs reach 45.0); pause s decoded (t20/m40/c5) {lib.covered(rg, 40*SR, 50*SR)/SR:.2f}, (t60/m75/c5) {lib.covered(rg60, 40*SR, 50*SR)/SR:.2f}; ranges60 {lib.sec(rg60)}; retime pauses {ps}")
# sweep
best = []
for el in (0.05, 0.06, 0.07, 0.08, 0.1, 0.15):
    for gap in (0.2, 0.3, 0.34, 0.36, 0.38, 0.39):
        for n in range(2, 13):
            for start in (4.6, 4.8, 4.9, 4.98):
                cl = [(start + k * (el + gap), el, -42) for k in range(n)]
                if cl[-1][0] + el > 13.5: continue
                wav = scene(cl, P=14.0)
                out = {}
                for tag in ("main", "head"):
                    lr = lib.loud_runs_sec(tag, wav, 42.5, 53.9)
                    rg = lib.plan(tag, wav).ranges
                    out[tag] = (round(max((b for a, b in lr), default=0), 2), round(lib.covered(rg, 40 * SR, 54 * SR) / SR, 2))
                best.append((round(out["head"][0] - 45.0, 2), el, gap, n, start, out))
print("sweep cases", len(best))
print("max head extension past reach:", max(b[0] for b in best), [b for b in sorted(best, reverse=True)[:3]])
print("max main extension past reach:", max(b[5]["main"][0] - 45 for b in best))
more = [b for b in best if b[5]["head"][1] > b[5]["main"][1] + 0.01]
less = [b for b in best if b[5]["head"][1] < b[5]["main"][1] - 0.01]
print("head decodes more pause:", len(more), "max extra", max((b[5]["head"][1] - b[5]["main"][1] for b in more), default=0), "; less:", len(less))
for b in sorted(more, key=lambda b: -(b[5]["head"][1] - b[5]["main"][1]))[:5]: print("  ", b)
