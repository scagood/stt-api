import sys
from h import *
import fuzz
var = load("chunker_variant", S + "/rr71-scripts/chunker_variant.py")
fuzz.main_(int(sys.argv[2]), vers={"new": new, "variant": var, "old": old})
