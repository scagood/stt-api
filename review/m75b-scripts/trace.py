import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/m75b-scripts")
from lib import *
ch = MODS["head"][0]
def trace(rms, ratio, relisten, mode):
    loud = rms > ch.relative_gate(rms, ratio)
    todo = ch._runs_of(~loud, relisten)
    print(mode, "file gate dB", round(20*np.log10(ch.relative_gate(rms, ratio)),1), "todo", todo)
    while todo:
        start, end = todo.pop()
        part = rms[start:end]
        gate = max(ch.relative_gate(part, ratio), float(np.percentile(part, 10)) * ch._OVER_FLOOR)
        padded = np.pad(part, 2, mode="edge")
        heard = np.median(np.lib.stride_tricks.sliding_window_view(padded, 5), axis=1) > gate
        if mode == "prev":
            kept = heard if heard.sum() >= 25 else np.zeros_like(heard)
        else:
            near = np.zeros_like(heard)
            for a, b in ch._joined(ch.runs(heard), 20).tolist():
                if heard[a:b].sum() >= 25:
                    near[max(0, a - relisten): b + relisten] = True
            kept = heard & near
        R = lambda m: [(start + a, start + b) for a, b in ch.runs(m).tolist()]
        print(f"  run ({start},{end}) gate {20*np.log10(gate):.1f} heard {R(heard)} kept {R(kept)}")
        if kept.any():
            loud[start:end] = kept
            new = [(start + a, start + b) for a, b in ch._runs_of(~kept, relisten)]
            print("    -> todo +", new)
            todo.extend(new)
    return loud
import subprocess
