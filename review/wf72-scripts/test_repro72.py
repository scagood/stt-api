import numpy as np
import pytest
from types import SimpleNamespace

from parakeet_service import routes
from parakeet_service.config import TARGET_SR



from tests.test_stitch import _samples, _result, _two_pieces, _Redo, _request, _words, _prepared


def _setup():
    wav = np.arange(20 * TARGET_SR, dtype=np.float32)
    prepared = _two_pieces([wav[a:b] for a, b in _samples([(0.0, 13.0), (7.0, 20.0)])])
    prepared.speech = _samples([(0.0, 20.0)])
    return prepared


@pytest.mark.asyncio
async def test_f1_literal():
    prepared = _setup()
    first = _result([" One", " two", " three"], [0.5, 1.0, 6.0])
    stalled = routes._stalled(prepared, [first, _words(10, 3.0)])
    print("stretches(s)", {k: [(a / TARGET_SR, b / TARGET_SR) for a, b in v] for k, v in stalled.items()})
    again = _result([" One", " two", " uh", " three"], [0.5, 1.0, 3.0, 5.92])
    print("words_in", routes._words_in(again, 0, stalled[0]))
    worker = _Redo([again])
    out = await routes._redo_stalled(_request(worker), [prepared], [first, _words(10, 3.0)], "parakeet-v3:fp32")
    print("kept redo:", out[0] is again)
    assert out[0] is again


@pytest.mark.asyncio
async def test_f1_second_piece_grid_offset():
    # piece 1: range (10,20), window (7,20): window grid offset 3.0 s = 37.5 frames of 80 ms
    prepared = _setup()
    first0 = _words(10, 0.5)  # fine
    # window-relative, on 7.0 + 0.08k grid: One 10.52 abs(3.52), two 11.0 abs(4.0), three 16.04 abs (9.04), then a word a second
    first1 = _result([" One", " two", " three", " a", " b", " c"], [3.52, 4.0, 9.04, 10.04, 11.04, 12.04])
    stalled = routes._stalled(prepared, [first0, first1])
    print("stretches(s)", {k: [(a / TARGET_SR, b / TARGET_SR) for a, b in v] for k, v in stalled.items()})
    # redo from 10.0 on 10 + 0.08k grid: 'three' at 16.00 abs (one 'grid' earlier), plus one made-up word
    again = _result([" One", " two", " uh", " three", " a", " b", " c"], [0.48, 1.04, 3.04, 6.0, 7.04, 8.0, 9.04])
    print("words_in", routes._words_in(again, prepared.ranges[1][0], stalled[1]))
    worker = _Redo([again])
    out = await routes._redo_stalled(_request(worker), [prepared], [first0, first1], "parakeet-v3:fp32")
    print("kept redo:", out[1] is again)
    assert out[1] is again


@pytest.mark.asyncio
async def test_f1_leading_stretch():
    # piece 1's first in-range token at 14.04 abs: leading stretch (10, 14.04) is 4.04 s
    prepared = _setup()
    first0 = _words(10, 0.5)
    first1 = _result([" x", " a", " b", " c", " d", " e", " f"], [1.0, 7.04, 8.0, 9.04, 10.0, 11.04, 12.0])
    stalled = routes._stalled(prepared, [first0, first1])
    print("stretches(s)", {k: [(a / TARGET_SR, b / TARGET_SR) for a, b in v] for k, v in stalled.items()})
    again = _result([" uh", " a", " b", " c", " d", " e", " f"], [1.04, 3.92, 5.04, 6.0, 7.04, 8.0, 9.04])
    print("words_in", routes._words_in(again, prepared.ranges[1][0], stalled[1]))
    worker = _Redo([again])
    out = await routes._redo_stalled(_request(worker), [prepared], [first0, first1], "parakeet-v3:fp32")
    print("kept redo:", out[1] is again)
    assert out[1] is again


@pytest.mark.asyncio
async def test_f2():
    prepared = _setup()
    first = _result([" a", " b", " c", " d", " e", " f"], [0.3, 0.7, 1.1, 1.5, 1.9, 2.3])
    stalled = routes._stalled(prepared, [first, _words(10, 3.0)])
    print("stretches(s)", {k: [(a / TARGET_SR, b / TARGET_SR) for a, b in v] for k, v in stalled.items()})
    again = _result([" x", " y"], [8.0, 8.5])
    worker = _Redo([again])
    out = await routes._redo_stalled(_request(worker), [prepared], [first, _words(10, 3.0)], "parakeet-v3:fp32")
    print("kept redo:", out[0] is again)
    text, _s, _w = routes._stitch(prepared, out)
    print("text:", text)
    assert out[0] is again
