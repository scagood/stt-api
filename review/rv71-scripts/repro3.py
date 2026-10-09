import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, ".")
from lib import *
VERS["fix1"] = fix1; VERS["fix2"] = fix2
for model, sp, sec in [
  ("v2", [(3,45),(46,86)], 100),
  ("wh", [(3,25),(27,82)], 100),
  ("v2", [(3,41),(42,50)], 60),
  ("v2", [(3,13),(14,33.8)], 40),
  ("wh", [(3,25),(27,70)], 80),
]:
    print(model, sp, sec); show(model, sp, sec)
