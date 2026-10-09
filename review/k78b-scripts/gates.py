import numpy as np
import lib
def show(wav, ratio, relisten=150):
    c = lib.CM
    rms = c.frame_rms(wav)
    g = c.relative_gate(rms, ratio)
    loud0 = rms > g
    print(f" ratio {ratio} file gate {20*np.log10(g):.1f} dB")
    for a, b in c._runs_of(~loud0, relisten):
        part = rms[a:b]
        gate = max(c.relative_gate(part, ratio), float(np.percentile(part, 10)) * c._OVER_FLOOR)
        print(f"  stretch {a*0.02:.2f}-{b*0.02:.2f} gate {20*np.log10(gate):.1f} dB (mean term {20*np.log10(c.relative_gate(part, ratio)):.1f}, floor term {20*np.log10(np.percentile(part,10)*c._OVER_FLOOR):.1f})")
