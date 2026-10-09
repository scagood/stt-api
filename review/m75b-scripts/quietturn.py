import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/m75b-scripts")
from lib import *

def speechy(sec, level_db, rng, halting=False):
    """Syllabic noise: Hann-enveloped syllables 120-250 ms, words of 1-3, phrases of 2-7 words."""
    out = []
    n = 0
    tgt = int(sec * SR)
    words = []  # (start, end) samples of each word
    while n < tgt:
        nwords = int(rng.integers(1, 3)) if halting else int(rng.integers(2, 8))
        for w in range(nwords):
            ws = n
            for s in range(int(rng.integers(1, 4))):
                L = int(rng.uniform(0.12, 0.25) * SR)
                env = np.hanning(L)
                out.append(rng.standard_normal(L) * env * db(level_db) * 1.6)
                n += L
                g = int(rng.uniform(0.0, 0.05) * SR); out.append(np.zeros(g)); n += g
            words.append((ws, n))
            g = int((rng.uniform(0.45, 0.9) if halting else rng.uniform(0.05, 0.2)) * SR); out.append(np.zeros(g)); n += g
        g = int(rng.uniform(0.3, 0.9) * SR); out.append(np.zeros(g)); n += g
    x = np.concatenate(out)[:tgt]
    return x, [(a, min(b, tgt)) for a, b in words if a < tgt]

def scenario(level, floor, seed, halting=False, qsec=15):
    rng = np.random.default_rng(seed)
    q, words = speechy(qsec, level, rng, halting)
    first = rng.standard_normal(40 * SR) * db(-20)
    last = rng.standard_normal(40 * SR) * db(-20)
    wav = np.concatenate([first, q, last])
    wav = wav + rng.standard_normal(wav.size) * db(floor)
    off = first.size
    return wav.astype(np.float32), off, off + q.size, [(a + off, b + off) for a, b in words]

def covered(ranges, a, b):
    return sum(max(0, min(y, b) - max(x, a)) for x, y in ranges)

print("level floor halting | words fully decoded main/prev/head | words wholly in a retime pause main/prev/head")
for halting in (False, True):
    for floor in (-70, -60, -55):
        for level in (-36, -40, -44, -48):
            res = {n: [0, 0] for n in ("main", "prev", "head")}
            tot = 0
            for seed in range(4):
                wav, a, b, words = scenario(level, floor, seed, halting)
                tot += len(words)
                for n in ("main", "prev", "head"):
                    ch, rt = MODS[n]
                    rr = ch.plan_chunks(wav, **BOUNDS, context_sec=5.0).ranges
                    res[n][0] += sum(covered(rr, x, y) == y - x for x, y in words)
                    ps = [(int(p * SR), int(q * SR)) for p, q in rt.pauses(wav)]
                    res[n][1] += sum(any(p <= x and y <= q for p, q in ps) for x, y in words)
            print(f"{level:4d} {floor:4d} {halting!s:5s} | {tot:3d}: " + "/".join(str(res[n][0]) for n in res) + " | " + "/".join(str(res[n][1]) for n in res))
