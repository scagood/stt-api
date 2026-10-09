import sys
sys.path.insert(0, ".")
sys.path.insert(0, sys.argv[1])
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import quietwords as q
from parakeet_service import retime
for seed in range(20):
    wav, words, phrases, (qa, qb) = q.build(seed, -40, -55)
    ps = retime.pauses(wav)
    for a, b in words:
        m = (a + b) / 2
        hit = [(round(x, 2), round(y, 2)) for x, y in ps if x <= m <= y]
        if hit:
            dist = min(min(abs(a - pe), abs(ps_ - b)) for ps_, pe in phrases)
            print(f"seed {seed} word {a:.2f}-{b:.2f} in pause {hit}, nearest phrase edge {dist:.2f}s, quiet turn ends {qb:.2f}")
