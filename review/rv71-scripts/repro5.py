import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, ".")
from lib import *
VERS["fix1"] = fix1; VERS["fix3"] = fix3
for m, sp, sec in [("v2", [(3,45),(46,85.8),(86.6,95)], 100), ("v3", [(3,25),(27,156)], 170), ("wh", [(3,25),(27,82),(83,90)], 100)]:
    print(m, sp); show(m, sp, sec)
