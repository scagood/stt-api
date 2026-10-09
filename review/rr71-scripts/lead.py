from h import *
import fuzz
lv = load("chunker_lead", S + "/rr71-scripts/chunker_lead.py")
for sp in ([(3, 41), (42, 50)], [(3, 41)], [(2, 21.5), (22.5, 30)]):
    print(sp, "new", run(new, sp, 60), "lead-variant", run(lv, sp, 60))
fuzz.main_(3000, bs=("v2",), vers={"new": new, "leadvar": lv})
