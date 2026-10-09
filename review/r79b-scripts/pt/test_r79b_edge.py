import pytest
from parakeet_service import routes
from tests.test_stitch import _Redo, _filler, _heard, _one_piece, _request, _samples


@pytest.mark.asyncio
@pytest.mark.parametrize("early", [0.32, 0.4, 0.48, 0.56])
async def test_word_a_redo_times_early_as_its_input_ends(early):
    # The piece skips 20-28 s and resumes at "Rome" (28 s): "Rome fell. [pause] Then Caesar ..."
    # Its redo (2 s past "Rome", so its input ends at 30 s) times the last words
    # before its input ends early, as the real model did (13.84 -> 13.36 s).
    before = _filler("a", 0.5, 19.5)
    after = [(" Rome", 28.0), (" fell.", 28.24), (" Then", 29.2), (" Caesar", 29.6)] + _filler("b", 30.4, 39.5)
    prepared = _one_piece(40.0, speech=_samples([(0.0, 40.0)]))
    middle = [(" the", 20.4), (" past", 24.0)]
    redo = _heard(17.82, [(" a38", 19.5)] + middle + [(" Rome", 28.0), (" fell.", 28.24), (" Then", 29.2 - early), (" Caesar", 29.6 - early)])
    results = await routes._redo_stalled(_request(_Redo([redo])), [prepared], [_heard(0.0, before + after)], "parakeet-v3:fp32")
    words = routes._stitch(prepared, results)[0].split()
    i = words.index("Rome")
    print(early, words[i - 3: i + 6])
    assert words.count("Then") == 1 and words.count("Caesar") == 1
