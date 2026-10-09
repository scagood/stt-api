import sys
sys.path.insert(0, ".")
sys.path.insert(0, sys.argv[1])
import repro
print("350ms yes @44.8 + 300ms breath @47:", repro.plan(repro.build(10, [(44.8, 0.35, -40), (47.0, 0.3, -42)]))[0])
