import asyncio, sys
sys.path.insert(0, ".")
sys.path.insert(0, "tests")
import conftest  # stubs onnxruntime
from test_stitch import _two_pieces, _samples, _result, _words, _Redo, _request
from parakeet_service import routes

async def run(first, again, label):
    prepared = _two_pieces()
    prepared.speech = _samples([(0.0, 20.0)])
    stalled = routes._stalled(prepared, first)
    worker = _Redo([again])
    out = await routes._redo_stalled(_request(worker), [prepared], list(first), "parakeet-v3:fp32")
    kept = any(o is again for o in out)
    print(f"{label}: stalled={ {k: [(a/16000, b/16000) for a, b in v] for k, v in stalled.items()} } redo_pieces={len(worker.pieces)} kept={kept}")
    return kept

async def main():
    # F1 trailing: piece 1, window 7, range (10,20). first decode words rel window
    heard = [3.0, 3.5, 4.0, 9.04, 9.5, 10.0, 11.0, 12.0]
    skipping = _result([f" w{i}" for i in range(3)] + [" three"] + [f" v{i}" for i in range(4)], heard)
    again = _result([f" w{i}" for i in range(3)] + [" uh", " three"] + [f" v{i}" for i in range(4)],
                    [0.0, 0.5, 1.0, 3.04, 6.0, 6.5, 7.0, 8.0, 9.0])
    await run([_words(10, 0.5), skipping], again, "F1 trailing (uh@13.04 + three@16.00)")
    # Control for trailing: uh@13.04 + um@14.0 (both well inside)
    again_c = _result([f" w{i}" for i in range(3)] + [" uh", " um", " three"] + [f" v{i}" for i in range(4)],
                    [0.0, 0.5, 1.0, 3.04, 4.0, 6.0, 6.5, 7.0, 8.0, 9.0])
    await run([_words(10, 0.5), skipping], again_c, "F1 trailing control (uh@13.04 + um@14.0)")
    # boundary: second word at 16.04-0.32=15.72 abs -> rel 5.72 ; and 15.64 (one frame earlier)
    for t in [5.80, 5.72, 5.64]:
        again_b = _result([f" w{i}" for i in range(3)] + [" uh", " um", " three"] + [f" v{i}" for i in range(4)],
                    [0.0, 0.5, 1.0, 3.04, t, 6.0, 6.5, 7.0, 8.0, 9.0])
        await run([_words(10, 0.5), skipping], again_b, f"F1 trailing boundary um@{10+t:.2f}")
    # F1 leading: stretch (10, 14.04). first decode piece 1: words at abs 9.5 (rel 2.5), then 14.04 (rel 7.04), 15..19
    lead = _result([" p", " a"] + [f" v{i}" for i in range(5)], [2.5, 7.04, 8.0, 9.0, 10.0, 11.0, 12.0])
    again_l = _result([" uh", " a"] + [f" v{i}" for i in range(5)], [1.04, 3.92, 5.0, 6.0, 7.0, 8.0, 9.0])
    await run([_words(10, 0.5), lead], again_l, "F1 leading (uh@11.04 + a@13.92)")
    again_lc = _result([" uh", " um", " a"] + [f" v{i}" for i in range(5)], [1.04, 2.5, 4.04, 5.0, 6.0, 7.0, 8.0, 9.0])
    await run([_words(10, 0.5), lead], again_lc, "F1 leading control (uh@11.04 + um@12.5)")
    # F2: piece 0
    stopped = _result([f" w{i}" for i in range(6)], [0.3, 0.7, 1.1, 1.5, 1.9, 2.3])
    await run([stopped, _words(10, 3.0)], _result([" x", " y"], [8.0, 8.5]), "F2 posted (6 lost, x,y gained)")
    await run([stopped, _words(10, 3.0)], _result([" w0", " w1", " w2", " w3", " x", " y"], [0.3, 0.7, 1.1, 1.5, 8.0, 8.5]), "F2 tie 6 vs 6")
    await run([stopped, _words(10, 3.0)], _result([" w0", " w1", " w2", " x", " y"], [0.3, 0.7, 1.1, 8.0, 8.5]), "F2 5 vs 6")
    # made-up word inside skipped speech
    first = _result([" a", " b", " c", " uh"], [0.3, 0.7, 1.1, 5.5])
    again11 = _result([f" r{i}" for i in range(11)], [0.3 + 0.85 * i for i in range(11)])
    await run([first, _words(10, 3.0)], again11, "made-up uh inside stretch, redo 11 real words")
    # Low new issue: loop of 12 'the'
    loop = _result([" the"] * 12, [0.25 * i for i in range(12)])
    redo9 = _result([" a", " b", " c", " d", " e", " f", " g", " h", " i"], [0.3, 1.0, 1.7, 2.4, 4.0, 5.0, 6.0, 7.0, 8.0])
    await run([loop, _words(10, 3.0)], redo9, "Low: 12x 'the' loop vs 9 real words")

asyncio.run(main())

# nit: empty aligner_quantization
from starlette.datastructures import FormData
from types import SimpleNamespace
from fastapi import HTTPException
async def nit():
    async def form(*pairs):
        return FormData(list(pairs))
    for pairs in [[("aligner", "wav2vec2-base-960h"), ("aligner_quantization", "")],
                  [("aligner", "wav2vec2-base-960h"), ("aligner_quantization", "  ")],
                  [("aligner", "mms-300m-forced-aligner"), ("aligner_quantization", "int8")],
                  [("aligner", "wav2vec2-base-960h:int8"), ("aligner_quantization", " FP16 ")],
                  [("aligner", ""), ("aligner_quantization", "fp32")],
                  [("aligner_quantization", "fp32")]]:
        try:
            await routes._no_aligner_quantization(SimpleNamespace(form=lambda p=pairs: form(*p)))
            print(pairs, "-> no error")
        except HTTPException as e:
            print(pairs, "->", e.status_code, e.detail)
            suggested = e.detail.split("send aligner=")[1].rsplit(" instead", 1)[0]
            try:
                print("   _named(suggested) ->", routes._named(suggested, "aligner"))
            except HTTPException as e2:
                print("   _named(suggested) -> 400:", e2.detail)
asyncio.run(nit())
