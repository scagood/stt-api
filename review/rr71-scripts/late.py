from h import *
import fuzz
lt = load("chunker_late", S + "/rr71-scripts/chunker_late.py")
vr = load("chunker_variant", S + "/rr71-scripts/chunker_variant.py")
fuzz.main_(3000, bs=("v2",), vers={"new": new, "fits-only": vr, "latest": lt})
