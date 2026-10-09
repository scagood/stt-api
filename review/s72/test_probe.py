import numpy as np, pytest
from types import SimpleNamespace
from parakeet_service import routes
from parakeet_service.config import TARGET_SR
import test_stitch as ts

async def _run(first, again):
    wav = np.arange(20 * TARGET_SR, dtype=np.float32)
    prepared = ts._two_pieces([wav[a:b] for a, b in ts._samples([(0.0, 13.0), (7.0, 20.0)])])
    prepared.speech = ts._samples([(0.0, 20.0)])
    results = [first, ts._words(10, 3.0)]
    print("stalled:", {k: [(a/TARGET_SR, b/TARGET_SR) for a, b in v] for k, v in routes._stalled(prepared, results).items()})
    out = await routes._redo_stalled(ts._request(ts._Redo([again])), [prepared], results, "parakeet-v3:fp32")
    return out[0] is again

@pytest.mark.asyncio
async def test_jitter_counts_as_new_word():
    first = ts._result([" One", " two", " three"], [0.5, 1.0, 6.0])
    # same words, 'three' one 80 ms frame earlier, plus ONE made-up word in the gap
    again = ts._result([" One", " two", " uh", " three"], [0.5, 1.0, 3.0, 5.92])
    assert await _run(first, again)  # kept on 1 genuinely new word

@pytest.mark.asyncio
async def test_redo_that_loses_words_is_kept():
    first = ts._result([" a", " b", " c", " d", " e", " f"], [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])  # stops at 3 s
    again = ts._result([" x", " y"], [8.0, 8.5])  # hears 2 words late, loses all six
    assert await _run(first, again)
