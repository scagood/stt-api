import sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

root = sys.argv[1]
mode = sys.argv[2]
sys.path.insert(0, root)
sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/wf75-scripts")
from parakeet_service import chunker  # noqa: E402

chunker.VAD = "volume"
chunker.VAD_GATE_DB = None
SR = chunker.TARGET_SR
_runs = chunker._runs


def smooth(part, k, how):
    if part.size < k:
        k = part.size
    pad = k // 2
    padded = np.pad(part, (pad, k - 1 - pad), mode="edge")
    win = sliding_window_view(padded, k)
    return np.median(win, axis=1) if how == "median" else win.min(axis=1)


def loud_frames(rms, ratio, relisten):
    loud = rms > max(chunker._GATE_FLOOR, float(rms.mean()) * ratio)
    todo = _runs(~loud, relisten)
    while todo:
        start, end = todo.pop()
        part = rms[start:end]
        gate = max(chunker._GATE_FLOOR, float(part.mean()) * ratio, float(np.percentile(part, 10)) * chunker._OVER_FLOOR)
        heard = smooth(part, chunker._SYLLABLE, mode) > gate
        if heard.any():
            loud[start:end] = heard
            todo.extend((start + a, start + b) for a, b in _runs(~heard, relisten))
    return loud


if mode != "pr":
    chunker.loud_frames = loud_frames


def db(x):
    return 10 ** (x / 20)


def build(pause_sec, noise_sec, noise_db, seed=0):
    rng = np.random.default_rng(seed)
    total = 40 + pause_sec + 40
    wav = rng.standard_normal(int(total * SR)) * db(-55)
    wav[: 40 * SR] = rng.standard_normal(40 * SR) * db(-20)
    wav[(40 + pause_sec) * SR:] = rng.standard_normal(40 * SR) * db(-20)
    a = int((40 + pause_sec / 2) * SR)
    n = int(noise_sec * SR)
    wav[a: a + n] = rng.standard_normal(n) * db(noise_db)
    return wav.astype(np.float32)


def _bursts(level_db, gap_db, bursts=3, burst_sec=1.0, gap_sec=1.0):
    rng = np.random.default_rng(0)
    tone = np.sin(2 * np.pi * 220 * np.arange(int(burst_sec * SR)) / SR) * np.sqrt(2) * 10 ** (level_db / 20)
    gap = rng.standard_normal(int(gap_sec * SR)) * 10 ** (gap_db / 20)
    return np.concatenate([part for _ in range(bursts) for part in (gap, tone)] + [gap]).astype(np.float32)


def _turn(seconds, level_db, floor_db=-70):
    return _bursts(level_db, floor_db, bursts=round(seconds / 0.35), burst_sec=0.25, gap_sec=0.1)


for name, pause, nsec, ndb in [("a click", 10, 0.01, -30), ("a2 knock", 10, 0.03, -35), ("b breath", 5, 0.3, -42), ("loud 10ms click -20", 10, 0.01, -20), ("60ms knock -35", 10, 0.06, -35)]:
    plan = chunker.plan_chunks(build(pause, nsec, ndb), target_sec=60.0, max_sec=75, context_sec=0)
    print(mode, name, [(round(a / SR, 2), round(b / SR, 2)) for a, b in plan.ranges])

for quiet_db in (-34, -44):
    first, quiet, last = _turn(50, -20), _turn(10, quiet_db), _turn(50, -20)
    plan = chunker.plan_chunks(np.concatenate([first, quiet, last]), target_sec=60.0, max_sec=75.0, context_sec=5.0)
    start, end = first.size, first.size + quiet.size
    cov = sum(max(0, min(b, end) - max(a, start)) for a, b in plan.ranges)
    print(mode, "quiet turn", quiet_db, cov == quiet.size)
# a short quiet word ("yes") in a long pause: 300 ms at -40 in -55 room tone
