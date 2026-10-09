import pytest
from parakeet_service import routes
from parakeet_service.config import TARGET_SR
from tests import test_stitch
import importlib.util, os
_spec = importlib.util.spec_from_file_location("repro72", os.path.join(os.path.dirname(__file__), "test_repro72.py"))
r = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(r)

_orig_stalled = routes._stalled
_first = {}

def _words_range(result, origin, low, high):
    info = routes._extract(result)
    starts = [origin + int(info["timestamps"][first] * TARGET_SR) for _w, first, _l in routes._word_spans(info["tokens"])]
    return sum(low <= at < high for at in starts)

def stalled(prepared, results):
    out = _orig_stalled(prepared, results)
    for i in out:
        (s, e), w = prepared.ranges[i], prepared.windows[i]
        _first[i] = _words_range(results[i], w[0], s, e)
    return out

orig_words_in = routes._words_in
def words_in(result, origin, stretches):
    # find which piece: by origin
    n = orig_words_in(result, origin, stretches)
    for i, cnt in _first.items():
        pass
    return n

@pytest.fixture(autouse=True)
def fix(monkeypatch):
    _first.clear()
    monkeypatch.setattr(routes, "_stalled", stalled)
    def wi(result, origin, stretches):
        # piece index lookup by stretches' range: compare with range in prepared via origin
        n = orig_words_in(result, origin, stretches)
        idx = [i for i in _first][0] if len(_first) == 1 else None
        total = _words_range(result, origin, origin, origin + 10**12)
        return n if total >= _first[idx] else 0
    monkeypatch.setattr(routes, "_words_in", wi)


@pytest.mark.asyncio
async def test_f2_fixed_drops():
    with pytest.raises(AssertionError):
        await r.test_f2()

test_a_piece_that_stops_short_is_decoded_again_without_context = test_stitch.test_a_piece_that_stops_short_is_decoded_again_without_context
test_a_redo_that_hears_no_more_words_is_dropped = test_stitch.test_a_redo_that_hears_no_more_words_is_dropped
test_a_piece_that_skips_speech_mid_way_is_decoded_again = test_stitch.test_a_piece_that_skips_speech_mid_way_is_decoded_again
