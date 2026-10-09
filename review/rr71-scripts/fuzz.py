import sys, collections
from h import *
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
def layout():
    at = rng.uniform(0, 5); sp = []
    seconds = rng.uniform(80, 400)
    while True:
        r = rng.random()
        L = rng.uniform(2, 6) if r < .8 else rng.uniform(6, 22) if r < .95 else rng.uniform(22, 70)
        if at + L > seconds - rng.uniform(0, 4): break
        sp.append((at, at + L)); at += L
        r = rng.random()
        at += rng.uniform(0.25, 2.0) if r < .9 else rng.uniform(2, 3.2) if r < .95 else rng.uniform(3, 12)
    return sp, seconds
def metrics(p, total, b, M):
    om = int((b["max_sec"] - 2 * b["context_sec"]) * SR); maximum = int(b["max_sec"] * SR)
    r, w, segs = p.ranges, p.windows, p.speech
    for s, e in r:
        if e <= s: M["zero_len"] += 1
        if e - s > om: M["over_om"] += 1
        if not any(a < e and s < z for a, z in segs): M["silent"] += 1
    for (s, e), (ws, we) in zip(r, w):
        if not (0 <= ws <= s < e <= we <= total) or we - ws > maximum: M["bad_window"] += 1
    for a, z in segs:
        if sum(max(0, min(z, e) - max(a, s)) for s, e in r) < z - a: M["speech_lost"] += 1
    for i, (s, e) in enumerate(r[:-1]):
        if r[i + 1][0] != e: continue
        c = e
        for a, z in segs:
            if a < c < z:
                M["cut_in_speech"] += 1
                if z - a <= om: M["cut_in_fitting"] += 1
            if 0 <= a - c < int(0.05 * SR): M["cut_at_onset"] += 1
    M["chunks"] += len(r)
    tm = r[-1][1] - segs[-1][1]; lm = segs[0][0] - r[0][0]
    if segs[-1][1] < total - SR:
        if tm < int(0.5 * SR): M["trail<0.5"] += 1
        if tm <= 0: M["trail0"] += 1
    if segs[0][0] > SR and lm < int(0.5 * SR): M["lead<0.5"] += 1
def main_(N, bs=("v2", "v3", "wh"), vers=VERS):
    for b in bs:
        Ms = {v: collections.Counter() for v in vers}
        for _ in range(N):
            sp, seconds = layout()
            if not sp: continue
            spS = [tuple(int(x * SR) for x in s) for s in sp]; total = int(seconds * SR)
            for v, mod in vers.items():
                try:
                    p = plan(mod, spS, total, BOUNDS[b])
                except Exception as ex:
                    Ms[v]["EXC " + type(ex).__name__] += 1; continue
                if p.speech: metrics(p, total, BOUNDS[b], Ms[v])
        keys = sorted(set().union(*[m.keys() for m in Ms.values()]))
        print(b, " ".join(f"{k}=" + "/".join(str(Ms[v][k]) for v in vers) for k in keys))
if __name__ == "__main__":
    main_(int(sys.argv[2]) if len(sys.argv) > 2 else 3000)
