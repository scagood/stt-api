import sys
sys.path.insert(0, ".")
from parakeet_service import retime
sys.path.insert(0, sys.argv[1])
import repro
for s in ([(43, 0.25, -42), (48, 0.25, -42), (53, 0.25, -42)], []):
    wav = repro.build(15, s)
    print([(round(float(a), 2), round(float(b), 2)) for a, b in retime.pauses(wav) if b > 39 and a < 56])
