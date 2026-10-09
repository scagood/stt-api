import sys
sys.argv = sys.argv[:1]
exec(open(sys.path[0] + "/repro.py").read().split("cases = {")[0])
ev = [(41 + 0.27 * k, 0.01, -30, "n") for k in range(30)]
print("typing 30x10ms -30 (10s)", plan(make(10, ev)))
