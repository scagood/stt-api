import sys; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, ".")
from lib import *
VERS["fix1"] = fix1
for sp, sec in [([(20,24),(28.44,46.3),(48.26,67.72),(69.5,87.92),(88.78,92)], 100),
                ([(28.44,46.3),(48.26,67.72),(69.5,87.92),(88.78,92)], 100)]:
    print(sp); show("v2", sp, sec)
