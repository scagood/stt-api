import sys
sys.path.insert(0, ".")
sys.path.insert(0, sys.argv[1])
import repro
# thirty 10 ms clicks at -30 dBFS, 0.27 s apart, in a 10 s pause
sounds = [(41.0 + 0.27 * i, 0.01, -30) for i in range(30)]
print("typing:", repro.plan(repro.build(10, sounds))[0])
