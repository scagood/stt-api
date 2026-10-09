"""Interview turn with real speech: a quieter talker (a LibriVox ch.3 clip, X dB under) just before (lead) or after (tail)
36-39 s of LibriVox ch.5 with its pauses cut to 0.2 s (one VAD segment), room tone at the file edges, optional noise bed.
Real volume VAD, v2 ctx5. The quiet clip's aligned words inside a range: main vs head (and an alternative module).
usage: margin_real.py N SIDE UNDER_DB [NOISE_DBFS]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import Counter
from common import MAIN, HEAD, SR, plan, load_local

DL = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/r80a-dl"
WDL = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/k80a-dl"
FIX = load_local(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.environ.get("FIXMOD", "chunker_fixA.py")), "fix")
N, side, under = int(sys.argv[1]), sys.argv[2], float(sys.argv[3])
noise = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "none" else None
rt60 = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
c5 = np.fromfile(f"{DL}/holmes_05.f32", dtype=np.float32); w5 = np.load(f"{WDL}/words_05.npy")
c3 = np.fromfile(f"{DL}/holmes_03.f32", dtype=np.float32); c3 = c3 - c3.mean(); w3 = np.load(f"{WDL}/words_03.npy")
# ch.5 with pauses over 0.2 s cut to 0.2 s
keep = np.ones(c5.size, bool)
for (s0, e0), (s1, e1) in zip(w5[:-1], w5[1:]):
    if s1 - e0 > 0.2:
        keep[int((e0 + 0.1) * SR): int((s1 - 0.1) * SR)] = False
sq = c5[keep]
def level(x):
    r = MAIN.frame_rms(x); return float(np.sqrt(np.mean(r[r > np.percentile(r, 50)] ** 2)))
L5 = level(sq)
T = Counter()
rng = np.random.default_rng(int(under) * 7 + (1 if side == "lead" else 2) + (0 if noise is None else 1000))
for i in range(N):
    Ld = rng.uniform(36, 39); a = int(rng.uniform(60, sq.size / SR - 60) * SR); main_part = sq[a: a + int(Ld * SR)]
    # quiet clip: starts at a word start in ch.3, 1.5-4 s long, ends at a word end
    k = int(rng.integers(100, len(w3) - 100)); qs = w3[k, 0]; qe_target = qs + rng.uniform(1.5, 4.0)
    j = k
    while j + 1 < len(w3) and w3[j + 1, 1] <= qe_target:
        j += 1
    qe = w3[j, 1] + 0.05
    clip = c3[int((qs - 0.03) * SR): int(qe * SR)].copy()
    qwords = [(s - (qs - 0.03), e - (qs - 0.03)) for s, e in w3[k: j + 1]]
    if rt60 > 0:  # off-mic in a room: direct sound plus an exponentially decaying diffuse tail
        n_ir = int(rt60 * SR)
        ir = rng.standard_normal(n_ir) * np.exp(-6.9 * np.arange(n_ir) / n_ir)
        ir *= 1.0 / np.sqrt(np.sum(ir ** 2)); ir[0] += 1.0
        m = 1 << int(np.ceil(np.log2(clip.size + n_ir)))
        clip = np.fft.irfft(np.fft.rfft(clip, m) * np.fft.rfft(ir, m), m)[: clip.size + n_ir].astype(np.float32)
    clip *= L5 * 10 ** (-under / 20) / max(level(clip), 1e-9)
    edge = rng.uniform(0.1, 1.0); turn = rng.uniform(0.05, 0.3); tail_room = rng.uniform(0.2, 2.0)
    room = lambda n: (rng.standard_normal(n) * 10 ** (-65 / 20)).astype(np.float32)
    if side == "lead":
        parts = [room(int(edge * SR)), clip, room(int(turn * SR)), main_part, room(int(tail_room * SR))]
        off = int(edge * SR) / SR
    else:
        parts = [room(int(edge * SR)), main_part, room(int(turn * SR)), clip, room(int(tail_room * SR))]
        off = (int(edge * SR) + main_part.size + int(turn * SR)) / SR
    wav = np.concatenate(parts).astype(np.float32)
    if noise is not None:
        wav += (rng.standard_normal(wav.size) * 10 ** (noise / 20)).astype(np.float32)
    W = [(int((s + off) * SR), int((e + off) * SR)) for s, e in qwords]
    res = {}
    for nm, mod in (("m", MAIN), ("h", HEAD), ("f", FIX)):
        p = plan(mod, wav, "v2 ctx5", gate=None)
        cov = np.zeros(wav.size, bool)
        for x, y in p.ranges: cov[x:y] = True
        res[nm] = [bool(cov[x:y].all()) for x, y in W]
        res[nm + "p"] = p
    T["files"] += 1; T["words"] += len(W); T["split"] += len(res["mp"].ranges) > 1
    T["heard"] += sum(any(s < y and e > x for s, e in res["mp"].speech) for x, y in W)
    for nm in "mhf":
        T[nm] += sum(res[nm])
    lh = sum(x and not y for x, y in zip(res["m"], res["h"])); lf = sum(x and not y for x, y in zip(res["m"], res["f"]))
    T["h_files"] += lh > 0; T["h_words"] += lh; T["f_files"] += lf > 0; T["f_words"] += lf
    if lh and os.environ.get("DUMP") and T["h_files"] <= int(os.environ.get("DUMP")):
        dd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dump"); os.makedirs(dd, exist_ok=True)
        tag = f"{side}_{int(under)}_{'n' if noise is None else int(-noise)}_{i}"
        wav.tofile(f"{dd}/{tag}.f32")
        import json
        json.dump(dict(main_ranges=res["mp"].ranges, main_windows=res["mp"].windows, head_ranges=res["hp"].ranges, head_windows=res["hp"].windows,
                       lost=[(x, y) for (x, y), p, q in zip(W, res["m"], res["h"]) if p and not q], words=W, speech=res["mp"].speech), open(f"{dd}/{tag}.json", "w"))
    if lh and T["h_files"] <= 2:
        print(f"  EX {i}: clip {off:.2f}-{off+clip.size/SR:.2f}s VAD {[(round(s/SR,2), round(e/SR,2)) for s,e in res['mp'].speech][:3]}... main {[(round(x/SR,2), round(y/SR,2)) for x,y in res['mp'].ranges]} head {[(round(x/SR,2), round(y/SR,2)) for x,y in res['hp'].ranges]} lost {[(round(x/SR,2), round(y/SR,2)) for (x,y),p,q in zip(W,res['m'],res['h']) if p and not q]}", flush=True)
print(f"real speech {side}, quiet talker {under:.0f} dB under, noise {noise}, rt60 {rt60}: files {T['files']} (split {T['split']}), words {T['words']} (VAD heard {T['heard']}), "
      f"in a range main {T['m']} head {T['h']} fixA {T['f']}; head fewer in {T['h_files']} files ({T['h_words']} words); fixA fewer in {T['f_files']} files ({T['f_words']} words)", flush=True)
