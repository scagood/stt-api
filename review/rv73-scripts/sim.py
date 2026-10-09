import sys, types, random, re
from types import SimpleNamespace
import numpy as np
try:
    import onnxruntime
except ImportError:
    o=types.ModuleType('onnxruntime'); o.get_available_providers=lambda:['CPUExecutionProvider']; sys.modules['onnxruntime']=o; sys.modules['onnx_asr']=types.ModuleType('onnx_asr')
from parakeet_service import routes
from parakeet_service.config import TARGET_SR
F=0.08
def prep(ranges, windows):
    r=[(int(round(s*TARGET_SR)),int(round(e*TARGET_SR))) for s,e in ranges]
    w=[(int(round(s*TARGET_SR)),int(round(e*TARGET_SR))) for s,e in windows]
    return routes._PreparedAudio(waveform=None,ranges=r,windows=w,speech=[],pieces=[np.zeros(e-s,dtype=np.float32) for s,e in w],duration=ranges[-1][1])
def piece_times(truth, ws, we, rng):
    out=[]; last=-1
    for t in truth:
        if not (ws <= t < we - 0.05): out.append(None); continue
        k = int((t - ws)/F) + (1 if rng.random() < 0.3 else 0) - (1 if rng.random()<0.1 else 0)
        k = max(k, last); last = k
        out.append(k*F)
    return out
mode = sys.argv[1]  # mishear or drop
rng = random.Random(int(sys.argv[2]) if len(sys.argv)>2 else 1)
stats = {"lost":0,"dup":0,"lost_straddle_diff":0,"dup_straddle_diff":0,"lost_other":0,"dup_other":0,"lost_onlyone_ownrange":0,"dup_onlyone_ownrange":0,"lost_onlyone_otherrange":0,"dup_onlyone_otherrange":0}
N=4000
for trial in range(N):
    t=17.0; truth=[]
    while t < 23.0:
        t += rng.uniform(0.12, 0.6); truth.append(round(t,3))
    ws2 = 15.0 + rng.randrange(0, 2000)/TARGET_SR*8  # right window start offset in samples
    windows=[(0.0,25.0),(ws2,40.0)]
    ranges=[(0.0,20.0),(20.0,40.0)]
    L = piece_times(truth, 0.0, 25.0, rng); R = piece_times(truth, ws2, 40.0, rng)
    lw=[]; rw=[]; diff=set(); onlyone=set(); kept_t={}
    for idx,(a,b) in enumerate(zip(L,R)):
        nl = f"w{idx}"; nr = f"w{idx}"
        if abs(truth[idx]-20) < 1.2:
            if mode=="mishear" and rng.random()<0.1:
                if rng.random()<0.5: nl=f"m{idx}"
                else: nr=f"m{idx}"
                diff.add(idx)
            if mode=="drop" and rng.random()<0.1:
                if rng.random()<0.5: a=None
                else: b=None
                onlyone.add(idx); kept_t[idx]=(a,b)
        if a is not None: lw.append((" "+nl, a))
        if b is not None: rw.append((" "+nr, b))
    res=[SimpleNamespace(text="", tokens=[w for w,_ in p], timestamps=[x for _,x in p]) for p in (lw,rw)]
    text,_s,words = routes._stitch(prep(ranges,windows), res)
    counts={}
    for w in text.split():
        i=int(w[1:]); counts[i]=counts.get(i,0)+1
    for idx,tt in enumerate(truth):
        if abs(tt-20)>1.2: continue
        c=counts.get(idx,0)
        if c==1: continue
        a=L[idx]; b=R[idx]
        straddle = a is not None and b is not None and ((a < 20) != (ws2 + b < 20))
        if idx in onlyone:
            a,b = kept_t[idx]
            own = (a is not None and a < 20) or (b is not None and ws2 + b >= 20)
            kind = "onlyone_ownrange" if own else "onlyone_otherrange"
        else:
            kind = "straddle_diff" if (idx in diff and straddle) else "other"
        if c==0: stats["lost"]+=1; stats["lost_"+kind]+=1
        else: stats["dup"]+=c-1; stats["dup_"+kind]+=c-1
print(mode, stats)
