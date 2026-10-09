import pytest
from parakeet_service import routes
from parakeet_service.config import TARGET_SR
from tests import test_stitch
from tests.test_stitch import _result, _words, _Redo, _request
import importlib.util, os
_spec = importlib.util.spec_from_file_location("repro72", os.path.join(os.path.dirname(__file__), "test_repro72.py"))
r = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(r)


def _patched(result, origin, stretches):
    info = routes._extract(result)
    tail = int(routes._WORD_TAIL_SEC * TARGET_SR)
    starts = [origin + int(info["timestamps"][first] * TARGET_SR) for _w, first, _l in routes._word_spans(info["tokens"])]
    return sum(low <= at < high - tail for at in starts for low, high in stretches)


@pytest.fixture(autouse=True)
def fix(monkeypatch):
    monkeypatch.setattr(routes, "_words_in", _patched)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["test_f1_literal", "test_f1_second_piece_grid_offset", "test_f1_leading_stretch"])
async def test_fixed_drops(name):
    with pytest.raises(AssertionError):
        await getattr(r, name)()


test_a_piece_that_stops_short_is_decoded_again_without_context = test_stitch.test_a_piece_that_stops_short_is_decoded_again_without_context
test_a_redo_that_hears_no_more_words_is_dropped = test_stitch.test_a_redo_that_hears_no_more_words_is_dropped
test_a_piece_that_skips_speech_mid_way_is_decoded_again = test_stitch.test_a_piece_that_skips_speech_mid_way_is_decoded_again
test_no_piece_is_decoded_again_when_the_rest_is_silence = test_stitch.test_no_piece_is_decoded_again_when_the_rest_is_silence
