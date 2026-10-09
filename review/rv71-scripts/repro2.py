import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, ".")
from lib import *
# NEW1 counter-examples: slide to onset of an oversized phrase saves an in-speech cut
for model, sp, sec in [
  ("wh", [(3,25),(27,87)], 100),
  ("v2", [(3,15),(16,56)], 70),
  ("v3", [(3,25),(27,92)], 110),
  ("v2", [(3,45),(46,86)], 100),
]:
    print(model, sp, sec); show(model, sp, sec)
