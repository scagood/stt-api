"""Decode main's and head's first (lead) or last (tail) window with Parakeet v3 int8 (onnx_asr, CPU):
which words does each emit inside the stretch main decodes and head gives up? usage: pk_decode.py DUMPDIR"""
import os, sys, json, glob
import numpy as np
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import onnx_asr
SR = 16000
S = "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad"
m = onnx_asr.load_model("nemo-conformer-tdt", f"{S}/r79b-model", quantization="int8", providers=["CPUExecutionProvider"]).with_timestamps()


def words_of(res, offset):
    out, cur, t0 = [], "", None
    for tok, ts in zip(res.tokens, res.timestamps):
        if tok.startswith(" ") and cur:
            out.append((cur.strip(), t0)); cur, t0 = "", None
        if t0 is None:
            t0 = float(ts) + offset
        cur += tok
    if cur:
        out.append((cur.strip(), t0))
    return out


tot = dict(files=0, main_words_in_stretch=0, head_words_in_stretch=0, files_main_has=0)
for jf in sorted(glob.glob(f"{sys.argv[1]}/*.json")):
    d = json.load(open(jf)); wav = np.fromfile(jf[:-5] + ".f32", dtype=np.float32)
    lead = os.path.basename(jf).startswith("lead")
    idx = 0 if lead else -1
    mw, hw = d["main_windows"][idx], d["head_windows"][idx]
    mr, hr = d["main_ranges"][idx], d["head_ranges"][idx]
    stretch = (mr[0], hr[0]) if lead else (hr[1], mr[1])
    out = {}
    for nm, (a, b) in (("main", mw), ("head", hw)):
        r = m.recognize(wav[a:b])
        out[nm] = words_of(r, a / SR)
    inside = lambda ws: [w for w, t in ws if stretch[0] / SR <= t < stretch[1] / SR]
    mi, hi = inside(out["main"]), inside(out["head"])
    tot["files"] += 1; tot["main_words_in_stretch"] += len(mi); tot["head_words_in_stretch"] += len(hi); tot["files_main_has"] += bool(mi)
    print(f"{os.path.basename(jf)}: stretch given up {stretch[0]/SR:.2f}-{stretch[1]/SR:.2f}s, aligned quiet words lost {[(round(x/SR,2), round(y/SR,2)) for x,y in d['lost']]}", flush=True)
    print(f"   main emits there: {mi} | head emits there: {hi}", flush=True)
print(tot)
