"""Main vs #80 chunk plans on synthetic word-structured audio.

usage: synth.py SEED N GAP_DB GAP_MS_LO GAP_MS_HI [closures] [scales]
Layouts from the c74r fuzz generator (VAD segments patched); audio has words
(0.15-0.5 s, level base +- 4 dB, syllable dips 6-10 dB) separated by gaps
GAP_DB under the base level, plus room tone at -65 dBFS outside speech.
"""
import os, sys, random, importlib.util
from collections import Counter, defaultdict
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/r80b-head")
import parakeet_service, parakeet_service.config  # noqa

SR = parakeet_service.config.TARGET_SR


def load(name):
    spec = importlib.util.spec_from_file_location(f"parakeet_service._r80b_{name}", f"{D}/chunker_{name}.py")
    m = importlib.util.module_from_spec(spec); m.__package__ = "parakeet_service"; spec.loader.exec_module(m); return m


MAIN, PR = load("main"), load(os.environ.get("PRNAME", "pr"))
WORD_SD = float(os.environ.get("WORD_SD", "4"))
SYLL = os.environ.get("SYLL", "1") == "1"
GAPREL = os.environ.get("GAPREL", "0") == "1"
FRIC = float(os.environ.get("FRIC", "0"))
DIALOG_DB = float(os.environ.get("DIALOG_DB", "0"))
CONFIGS = {
    "v2 ctx5": dict(target=25.0, mx=30.0, ctx=5.0),
    "v2 ctx0": dict(target=25.0, mx=30.0, ctx=0.0),
    "v3 ctx5": dict(target=60.0, mx=75.0, ctx=5.0),
    "whisper": dict(target=25.0, mx=30.0, ctx=0.0),
}


def layout(rng, scale=1.0):
    t = rng.uniform(0, 4)
    segs = []
    for _ in range(rng.randint(1, 8)):
        L = rng.choice([rng.uniform(0.3, 6), rng.uniform(5, 25), rng.uniform(15, 50)]) * scale
        segs.append((t, t + L))
        t += L + rng.choice([rng.uniform(0.05, 1.0), rng.uniform(0.5, 2.9), rng.uniform(3, 8)])
    total = segs[-1][1] + rng.choice([0, rng.uniform(0, 1.5), rng.uniform(1, 6)])
    return segs, total


def synth(segs, total, nrng, gap_db, gap_ms, closures=0.0, base_db=-20.0, room_db=-65.0):
    """segs in samples. Returns wav (float32) and word spans (samples)."""
    env = np.full(total, 10 ** (room_db / 20), dtype=np.float32)
    words = []
    gap_amp = max(10 ** ((base_db - gap_db) / 20), 10 ** (room_db / 20))
    for s, e in segs:
        t = s
        nxt = base_db + nrng.normal(0, WORD_SD)
        drng = np.random.default_rng(int(s) + 7)
        turns = [s]
        while turns[-1] < e:
            turns.append(turns[-1] + int(drng.uniform(1.5, 6.0) * SR))
        off = lambda x: DIALOG_DB * ((np.searchsorted(turns, x, side="right") - 1) % 2) if DIALOG_DB else 0.0
        while t < e:
            wl = int(nrng.uniform(0.15, 0.5) * SR)
            we = min(e, t + wl)
            n = we - t
            cur_db = nxt - off(t)
            nxt = base_db + nrng.normal(0, WORD_SD)
            lvl = 10 ** (cur_db / 20)
            if GAPREL:
                nxt_db = nxt - max(off(we), off(we + int(0.35 * SR)))
                gap_amp = max(10 ** ((min(cur_db, nxt_db) - gap_db) / 20), 10 ** (room_db / 20))
            shape = np.ones(n, dtype=np.float32)
            # syllables: 1-3, each dip between them 6-10 dB for 30-60 ms
            nsyl = nrng.integers(1, 4)
            for k in range(1, nsyl if SYLL else 1):
                c = t + n * k // nsyl - t
                w = int(nrng.uniform(0.03, 0.06) * SR)
                shape[max(0, c - w // 2): c + w // 2] = 10 ** (-nrng.uniform(6, 10) / 20)
            if closures and nrng.random() < closures and n > int(0.2 * SR):
                # a stop closure: 40-100 ms, 25-35 dB down, inside the word
                w = int(nrng.uniform(0.04, 0.10) * SR)
                c = int(nrng.uniform(0.3, 0.7) * n)
                shape[max(0, c - w // 2): c + w // 2] = 10 ** (-nrng.uniform(25, 35) / 20)
            if FRIC and n > int(0.3 * SR) and drng.random() < FRIC:
                # a fricative (s, sh): 120-250 ms of noise 12-18 dB under the vowel
                w = min(int(drng.uniform(0.12, 0.25) * SR), n - int(0.05 * SR))
                c0 = int(drng.uniform(0.0, 1.0) * (n - w))
                shape[c0: c0 + w] = 10 ** (-drng.uniform(12, 18) / 20)
            # 10 ms onset/offset ramps
            r = min(int(0.01 * SR), n // 2)
            if r:
                shape[:r] *= np.linspace(0.1, 1, r, dtype=np.float32)
                shape[n - r:] *= np.linspace(1, 0.1, r, dtype=np.float32)
            env[t:we] = lvl * shape
            words.append((t, we))
            ge = min(e, we + int(nrng.uniform(*gap_ms) / 1000 * SR))
            env[we:ge] = gap_amp
            t = ge
    wav = nrng.standard_normal(total, dtype=np.float32) * env
    return wav, words


def plan(mod, wav, segs, cfg):
    mod._speech_segments = lambda _w: segs
    p = mod.plan_chunks(wav, target_sec=cfg["target"], max_sec=cfg["mx"], min_sec=20.0, context_sec=cfg["ctx"])
    return p.ranges, p.windows


def forced_cuts(ranges, segs):
    return [b for (a, b), (c, d) in zip(ranges, ranges[1:]) if b == c and any(s < b < e for s, e in segs)]


def in_word(x, starts, words):
    i = np.searchsorted(starts, x, side="right") - 1
    return i >= 0 and words[i][0] < x < words[i][1]


def cover(ranges, total):
    m = np.zeros(total, dtype=bool)
    for a, b in ranges:
        m[a:b] = True
    return m


def metrics(ranges, windows, segs, total, cfg, words, starts):
    own = int((cfg["mx"] - 2 * cfg["ctx"]) * SR); mx = int(cfg["mx"] * SR)
    bad = []
    if any(not (0 <= a < b <= total) for a, b in ranges): bad.append("outside")
    if any(b > c for (a, b), (c, d) in zip(ranges, ranges[1:])): bad.append("overlap")
    if any(b - a > own for a, b in ranges): bad.append("range>own")
    if any(b - a > mx for a, b in windows): bad.append("window>max")
    if any(not (wa <= a and b <= wb) for (a, b), (wa, wb) in zip(ranges, windows)): bad.append("window!>=range")
    fc = forced_cuts(ranges, segs)
    lost = sum((e - s) - sum(max(0, min(b, e) - max(a, s)) for a, b in ranges) for s, e in segs)
    return dict(
        n=len(ranges), forced=len(fc), inword=sum(in_word(x, starts, words) for x in fc), fc=fc,
        short=sum(1 for a, b in ranges if b - a < 2 * SR),
        silent=sum(1 for a, b in ranges if not any(s < b and e > a for s, e in segs)),
        lost=lost, bad=bad, maxwin=max((b - a for a, b in windows), default=0),
        minlen=min((b - a for a, b in ranges), default=0),
    )


def run(seed, N, gap_db, gap_ms, closures=0.0, scales=(1.0, 1.6), configs=None, verbose=True):
    out = {}
    for cname in (configs or CONFIGS):
        cfg = CONFIGS[cname]
        for scale in scales:
            rng = random.Random(f"k80a|{seed}|{cname}|{scale}")
            nrng = np.random.default_rng(abs(hash((seed, cname, scale, gap_db, gap_ms, closures))) % 2**32)
            T = Counter(); ex = []
            moves = []
            for _ in range(N):
                segs_s, total_s = layout(rng, scale)
                if total_s <= cfg["mx"]:
                    continue
                total = int(round(total_s * SR))
                segs = [(int(round(a * SR)), int(round(b * SR))) for a, b in segs_s]
                wav, words = synth(segs, total, nrng, gap_db, gap_ms, closures)
                starts = np.array([w[0] for w in words])
                rm, wm = plan(MAIN, wav, segs, cfg)
                rp, wp = plan(PR, wav, segs, cfg)
                a = metrics(rm, wm, segs, total, cfg, words, starts)
                b = metrics(rp, wp, segs, total, cfg, words, starts)
                T["layouts"] += 1
                for k in ("n", "forced", "inword", "short", "silent"):
                    T["main_" + k] += a[k]; T["pr_" + k] += b[k]
                T["main_lost_s"] += a["lost"] / SR; T["pr_lost_s"] += b["lost"] / SR
                T["main_bad"] += bool(a["bad"]); T["pr_bad"] += bool(b["bad"])
                T["pr_maxwin_s"] = max(T["pr_maxwin_s"], b["maxwin"] / SR)
                for k in ("n", "forced", "inword", "short", "silent", "lost"):
                    if b[k] > a[k]: T["worse_" + k] += 1
                    if b[k] < a[k]: T["better_" + k] += 1
                if b["minlen"] < a["minlen"] - 1: T["pr_minlen_shorter"] += 1
                if b["inword"] > a["inword"] and len(ex) < 3:
                    ex.append((segs_s, total_s))
                # paired movement
                if len(a["fc"]) == len(b["fc"]):
                    for x, y in zip(a["fc"], b["fc"]):
                        moves.append(abs(y - x) / SR)
                        wx, wy = in_word(x, starts, words), in_word(y, starts, words)
                        T[f"pair_{'W' if wx else 'g'}->{'W' if wy else 'g'}"] += 1
                else:
                    T["fc_count_differs"] += 1
                # coverage: audio in main's ranges not in #80's (and windows)
                cm, cp = cover(rm, total), cover(rp, total)
                drop = cm & ~cp
                if drop.any():
                    T["drop_layouts"] += 1
                    T["drop_s"] += drop.sum() / SR
                    T["drop_max_s"] = max(T["drop_max_s"], drop.sum() / SR)
                    # how far from VAD speech is the dropped audio (min over dropped samples)
                    idx = np.flatnonzero(drop)
                    dist = min(min(abs(i - s), abs(i - e)) for i in (idx[0], idx[-1]) for s, e in segs) / SR
                    T["drop_nearest_speech_min_s"] = dist if T["drop_layouts"] == 1 else min(T["drop_nearest_speech_min_s"], dist)
                    T["drop_inside_speech"] += int(any(drop[s:e].any() for s, e in segs))
                gm, gp = cover(wm, total), cover(wp, total)
                if (gm & ~gp).any():
                    T["windrop_layouts"] += 1
                    T["windrop_s"] += (gm & ~gp).sum() / SR
            mv = np.array(moves) if moves else np.zeros(0)
            T["moved_sum"] += float(mv.sum()); T["moved_n"] += mv.size; T["moved_zero"] += int((mv < 1 / SR).sum())
            T["moved_max_s"] = max(T["moved_max_s"], float(mv.max()) if mv.size else 0.0)
            out[(cname, scale)] = (T, ex)
            if verbose:
                report(cname, scale, T)
    return out


def report(cname, scale, T):
    if True:
            if True:
                T["moved_mean_s"] = T["moved_sum"] / max(1, T["moved_n"]); T["moved_zero_share"] = T["moved_zero"] / max(1, T["moved_n"])
                f = lambda k: T[k]
                print(f"{cname} x{scale} L={f('layouts')} pieces {f('main_n')}->{f('pr_n')} forced {f('main_forced')}->{f('pr_forced')} "
                      f"inword {f('main_inword')} ({f('main_inword')/max(1,f('main_forced')):.0%}) -> {f('pr_inword')} ({f('pr_inword')/max(1,f('pr_forced')):.0%}) "
                      f"moved mean {T['moved_mean_s']:.2f}s unmoved {T['moved_zero_share']:.0%} | <2s {f('main_short')}->{f('pr_short')} "
                      f"silent {f('main_silent')}->{f('pr_silent')} lost {f('main_lost_s'):.1f}->{f('pr_lost_s'):.1f}s bad {f('main_bad')}->{f('pr_bad')} "
                      f"maxwin {T['pr_maxwin_s']:.2f} | worse: n {f('worse_n')} inword {f('worse_inword')} short {f('worse_short')} silent {f('worse_silent')} lost {f('worse_lost')} "
                      f"better inword {f('better_inword')} | pairs W->W {f('pair_W->W')} W->g {f('pair_W->g')} g->g {f('pair_g->g')} g->W {f('pair_g->W')} "
                      f"| drop layouts {f('drop_layouts')} ({f('drop_s'):.1f}s, max {T['drop_max_s']:.2f}s, nearest {T['drop_nearest_speech_min_s']:.2f}s, inside speech {f('drop_inside_speech')}) "
                      f"windrop {f('windrop_layouts')}", flush=True)


if __name__ == "__main__":
    seeds, N, gdb, lo, hi = [int(x) for x in sys.argv[1].split(",")], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    clos = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    scales = tuple(float(x) for x in sys.argv[7].split(",")) if len(sys.argv) > 7 else (1.0, 1.6)
    configs = sys.argv[8].split(",") if len(sys.argv) > 8 else None
    print(f"# seeds={seeds} N={N} gap_db={gdb} gap_ms={lo}-{hi} closures={clos}", flush=True)
    agg = {}
    for seed in seeds:
        for key, (T, ex) in run(seed, N, gdb, (lo, hi), clos, scales, configs, verbose=False).items():
            A = agg.setdefault(key, Counter())
            for k, v in T.items():
                if k.endswith("_max_s") or k == "pr_maxwin_s":
                    A[k] = max(A[k], v)
                elif k == "drop_nearest_speech_min_s":
                    A[k] = v if k not in A else min(A[k], v)
                else:
                    A[k] += v
            A["examples"] = A.get("examples", 0)
            agg.setdefault(("ex",) + key, []).extend(ex)
    tot = Counter()
    for key, A in agg.items():
        if key[0] == "ex":
            continue
        report(*key, A)
        for k in ("main_forced", "pr_forced", "main_inword", "pr_inword", "worse_inword", "better_inword", "worse_n", "worse_lost", "worse_short", "worse_silent", "pr_bad", "drop_layouts", "layouts"):
            tot[k] += A[k]
    print("TOTAL", dict(tot), flush=True)
    for key, ex in agg.items():
        if key[0] == "ex" and ex:
            print("EX-worse-inword", key[1:], ex[0])
