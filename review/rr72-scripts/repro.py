import asyncio, sys, os
sys.path.insert(0, os.getcwd())
import tests.conftest  # stubs onnx
from tests.test_stitch import _two_pieces, _samples, _result, _words, _Redo, _request
from parakeet_service import routes

def run(name, results, again, which):
    prepared = _two_pieces()
    prepared.speech = _samples([(0.0, 20.0)])
    orig = list(results)
    stalled = routes._stalled(prepared, results)
    worker = _Redo([again])
    out = asyncio.run(routes._redo_stalled(_request(worker), [prepared], results, "parakeet-v3:fp32"))
    kept = out[which] is again
    print(f"{name}: stalled={ {k: [(a/16000, b/16000) for a, b in v] for k, v in stalled.items()} } redo_kept={kept}")

# F1 trailing (review main repro): piece 1, window 7, range 10
first1 = _result([" w0"," w1"," w2"," three"," v0"," v1"," v2"," v3"], [3.0,3.5,4.0,9.04,9.5,10.0,11.0,12.0])
redo1 = _result([" w0"," w1"," w2"," uh"," three"," v0"," v1"," v2"," v3"], [0.0,0.5,1.0,3.04,6.0,6.5,7.0,8.0,9.0])
run("F1 trailing repro (expect rejected)", [_words(10, 0.5), first1], redo1, 1)

# F1 leading-stretch repro: stretch (10.0, 14.04); redo uh@11.04 + a@13.92; redo total >= first total
first2 = _result([" p0", " a", " b", " c", " d"], [1.0, 7.04, 8.0, 9.0, 10.0])  # p0@8.0 in context; a@14.04
redo2 = _result([" uh", " a", " b", " c", " d"], [1.04, 3.92, 5.0, 6.0, 7.0])
run("F1 leading repro (expect rejected)", [_words(10, 0.5), first2], redo2, 1)

# F1 control: genuine 2 new words well inside stretch still kept
redo2b = _result([" uh", " um", " a", " b", " c", " d"], [1.04, 2.5, 4.04, 5.0, 6.0, 7.0])
run("F1 control 2 genuine words (expect kept)", [_words(10, 0.5), first2], redo2b, 1)

# F2 review repro: piece 0 range(0,10) window(0,13)
stopped = _result([f" w{i}" for i in range(6)], [0.3,0.7,1.1,1.5,1.9,2.3])
run("F2 repro (expect rejected)", [stopped, _words(10, 3.0)], _result([" x", " y"], [8.0, 8.5]), 0)

# F2 tie: redo hears 6 words in range including 2 new -> equal count -> kept?
redo_tie = _result([f" w{i}" for i in range(4)] + [" x", " y"], [0.3,0.7,1.1,1.5,8.0,8.5])
run("F2 tie 6 vs 6 (>= keeps)", [stopped, _words(10, 3.0)], redo_tie, 0)
redo_less = _result([f" w{i}" for i in range(3)] + [" x", " y"], [0.3,0.7,1.1,8.0,8.5])
run("F2 one fewer 5 vs 6 (rejected)", [stopped, _words(10, 3.0)], redo_less, 0)

# Better redo rejected: first decode looped/hallucinated 12 words in 0-3 s then stalled 3.3-10;
# redo hears 4 words in 0-3 and 5 in the 7 s it skipped (9 < 12)
loop = _result([" the"] * 12, [0.25 * i for i in range(12)])
better = _result([" a", " b", " c", " d", " e", " f", " g", " h", " i"], [0.3, 1.0, 1.7, 2.4, 4.0, 5.0, 6.0, 7.0, 8.0])
run("first decode 12 looped words then stall; redo 4+5 real (count rule rejects)", [loop, _words(10, 3.0)], better, 0)

# First decode hallucinated a word inside the skipped speech: 3 real words 0.3-1.1, made-up "uh"@5.5, nothing else.
halluc = _result([" a", " b", " c", " uh"], [0.3, 0.7, 1.1, 5.5])
real = _result([" a", " b", " c"] + [f" r{i}" for i in range(8)], [0.3, 0.7, 1.1] + [2.0 + i for i in range(8)])
run("first decode made-up word inside stretch; redo 11 real (expect kept)", [halluc, _words(10, 3.0)], real, 0)
