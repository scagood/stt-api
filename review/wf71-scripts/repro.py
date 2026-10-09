from harness import *
cases = {
 "F1": ([(3,13),(14,33.8)], 40),
 "F2": ([(2,21.5),(22.5,30)], 40),
 "F3a": ([(3,40),(41,50)], 60),
 "F3b": ([(3,40)], 60),
 "test": ([(3,19.8),(20.6,39.5)], 45),
}
for k,(sp,tot) in cases.items():
    for name, mod in (("PR", pr), ("main", main)):
        r = run(mod, sp, tot)
        print(k, name, r, "cuts in speech:", cuts_in_speech(r, sp))
