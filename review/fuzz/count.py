import sys, subprocess, json, numpy as np
SR = 16000
rng = np.random.default_rng(5)
cases = []
for i in range(400):
    secs = rng.uniform(100, 600); L0, L1 = sorted(rng.uniform(0.2, 22, 2)); P1 = rng.uniform(0.4, 2.9)
    sp, at = [], rng.uniform(0, 3)
    while True:
        L = rng.uniform(L0, L1); 
        if at + L > secs: break
        sp.append((int(at*SR), int((at+L)*SR))); at += L + rng.uniform(0.4, P1)
    cases.append((sp, int(secs*SR)))
json.dump(cases, open(sys.argv[1], 'w'))
