import sys
sys.path.insert(0, ".")
sys.path.insert(0, sys.argv[1])
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import repro
# 20 s pause: quiet 0.8 s phrase at 45.0, a 0.3 s word starting 2.85 s after the phrase ends, nothing else
for gap in (2.6, 2.85, 2.95, 3.2):
    w0 = 45.8 + gap
    r = repro.plan(repro.build(20, [(45.0, 0.8, -40), (w0, 0.3, -40)]), 0)[0]
    print(f"word {w0:.2f}-{w0 + 0.3:.2f} ({gap} s after phrase): {r}")
