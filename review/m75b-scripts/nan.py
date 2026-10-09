import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/m75b-scripts")
from lib import *
import warnings; warnings.simplefilter("ignore")
w = build(20, [(45, 0.6, -40), (47, 0.3, -42)])
for label, mod in (("nan block in pause", lambda x: x.__setitem__(slice(50*SR, 50*SR+3200), np.nan)), ("inf sample", lambda x: x.__setitem__(52*SR, np.inf)), ("nan in turn", lambda x: x.__setitem__(10*SR, np.nan))):
    x = w.copy(); mod(x)
    for n in ("prev", "head"):
        try:
            print(label, n, plan(n, x)[:5], len(pauses(n, x)))
        except Exception as e:
            print(label, n, "EXC", type(e).__name__, e)
