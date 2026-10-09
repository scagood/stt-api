# Differential vs main: random quiet-speech pauses; words = quiet phrases and
# short words (breaths are not words). Per case: words kept whole / cut / dropped
# in plan_chunks ranges; and in retime.pauses, words a pause edge falls inside
# or that lie wholly inside a pause.
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/v78-scripts")
from lib import *
_argv = sys.argv if __name__ == '__main__' else [sys.argv[0], '0', '0']
seed = int(_argv[1]); N = int(_argv[2]); mode = _argv[3] if len(_argv) > 3 else "random"
ms = int(_argv[4]) if len(_argv) > 4 else 400
NAMES = _argv[5].split(",") if len(_argv) > 5 else ["main", "head"]
for n in NAMES: MODS[n][0].VAD_MIN_SILENCE_MS = ms
rng = np.random.default_rng(seed)

def gen(it):
    floor = float(rng.uniform(-62, -50)); P = float(rng.uniform(5, 40))
    words, sounds = [], []
    t = 40 + float(rng.uniform(0.2, 3))
    spk = floor + float(rng.uniform(12, 30))  # this pause's quiet speaker level
    halting = rng.random() < 0.25
    while t < 40 + P - 0.3:
        k = rng.random()
        if mode == "adv" and words and rng.random() < 0.3:
            # put a short word just across the last phrase's 3 s reach
            last = max(e for s, e in words if e - s >= 0.5) if any(e - s >= 0.5 for s, e in words) else None
            if last and last + 3.0 - 0.4 > t:
                t = last + 3.0 - float(rng.uniform(0.02, 0.4))
                if t >= 40 + P - 0.15: break
        if k < (0.1 if halting else 0.3):
            d = float(rng.uniform(0.5, 2.0)); lvl = spk + float(rng.uniform(-4, 3)); words.append((t, t + d))
        elif k < 0.7:
            d = float(rng.uniform(0.12, 0.45)); lvl = spk + float(rng.uniform(-8, 3)); words.append((t, t + d))
        else:
            d = float(rng.uniform(0.08, 0.45)); lvl = floor + float(rng.uniform(5, 17))
        d = min(d, 40 + P - t)
        if words and words[-1][0] == t: words[-1] = (t, t + d)
        sounds.append((t, d, lvl))
        g = rng.choice([rng.uniform(0.05, 0.45), rng.uniform(0.45, 1.5), rng.uniform(1.5, 5)], p=[0.4, 0.4, 0.2])
        t += d + float(g)
    wav = build(P, sounds, seed=it + 1000003 * seed, floor_db=floor)
    return wav, words, P

def score_plan(ranges, words):
    rs = [(a / SR, b / SR) for a, b in ranges]
    kept = cut = dropped = 0
    for w0, w1 in words:
        cov = sum(max(0.0, min(b, w1) - max(a, w0)) for a, b in rs)
        inner_edge = any(w0 + 0.021 < x < w1 - 0.021 for a, b in rs for x in (a, b))
        if cov <= 0.0: dropped += 1
        elif cov < (w1 - w0) - 0.021 or inner_edge: cut += 1
        else: kept += 1
    return kept, cut, dropped

def score_pauses(ps, words):
    bad = 0
    for w0, w1 in words:
        if any((w0 + 0.021 < a < w1 - 0.021) or (w0 + 0.021 < b < w1 - 0.021) or (a <= w0 + 0.021 and w1 - 0.021 <= b) for a, b in ps):
            bad += 1
    return bad

tot = {n: [0, 0, 0, 0, 0, 0] for n in NAMES}
worse = {n: [] for n in NAMES}; better = {n: 0 for n in NAMES}
for it in range(N):
    wav, words, P = gen(it)
    res = {}
    for n in NAMES:
        k0, c0, d0 = score_plan(plan(n, wav, 0.0).ranges, words)
        k5, c5, d5 = score_plan(plan(n, wav, 5.0).ranges, words)
        pb = score_pauses(pauses(n, wav), words)
        res[n] = (k0, c0, d0, k5, c5 + d5, pb)
        for i, v in enumerate((k0, c0, d0, k5, c5 + d5, pb)): tot[n][i] += v
    m = res[NAMES[0]]
    for n in NAMES[1:]:
        r = res[n]
        if r[1] + r[2] > m[1] + m[2] or r[4] > m[4] or r[5] > m[5] or r[0] < m[0] or r[3] < m[3]:
            worse[n].append((it, m, r))
        if r[1] + r[2] < m[1] + m[2] or r[5] < m[5]: better[n] += 1
if N: print(f"mode {mode} seed {seed} ms {ms}: {N} pauses")
for n in (NAMES if N else []):
    k0, c0, d0, k5, cd5, pb = tot[n]
    print(f"  {n:5s} ctx0 kept {k0} cut {c0} dropped {d0} | ctx5 kept {k5} cut+dropped {cd5} | retime words with a pause in them {pb}")
for n in (NAMES[1:] if N else []):
    print(f"  {n} better than {NAMES[0]} in {better[n]} cases, worse in {len(worse[n])}: {worse[n][:6]}")
