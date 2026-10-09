"""Repros against the PR's own test helpers (run with the worktree's tests on sys.path)."""
import pytest
from parakeet_service import routes
from tests.test_stitch import _Redo, _filler, _heard, _one_piece, _request, _samples, _two_pieces, _result


@pytest.mark.asyncio
@pytest.mark.parametrize("at", [19.76, 19.84, 19.92, 20.0])
async def test_split_word_in_margin(at):
    # The PR's own misheard-word test, with "standing" timed `at` instead of 19.76.
    before = _filler("a", 0.5, 18.5) + [(" under", 19.52), ("stand", 19.68), ("ing", 19.84)]
    after = [(" Rome", 28.0)] + _filler("b", 28.5, 39.5)
    prepared = _one_piece(40.0, speech=_samples([(0.0, 40.0)]))
    redo = _heard(18.16, [(" under", 19.52), (" standing", at), (" the", 20.4), (" past", 24.0), (" roam", 28.08)])
    results = await routes._redo_stalled(_request(_Redo([redo])), [prepared], [_heard(0.0, before + after)], "parakeet-v3:fp32")
    words = routes._stitch(prepared, results)[0].split()
    print(at, words[35:41])
    assert "standing" not in words


@pytest.mark.asyncio
@pytest.mark.parametrize("gap", [0.16, 0.24, 0.32])
async def test_into_heard_as_in_to(gap):
    # A word 1 s before the skip, which the redo hears as two.
    before = _filler("a", 0.5, 18.0) + [(" into", 18.4), (" the", 18.8), (" we", 19.2), (" were", 19.5)]
    after = [(" Rome", 28.0)] + _filler("b", 28.5, 39.5)
    prepared = _one_piece(40.0, speech=_samples([(0.0, 40.0)]))
    redo = _heard(17.82, [(" in", 18.4), (" to", 18.4 + gap), (" the", 18.8), (" we", 19.2), (" were", 19.5), (" past", 24.0), (" ages", 24.5), (" Rome", 28.0)])
    results = await routes._redo_stalled(_request(_Redo([redo])), [prepared], [_heard(0.0, before + after)], "parakeet-v3:fp32")
    words = routes._stitch(prepared, results)[0].split()
    print(gap, words[33:42])
    assert words.count("to") == 0
