import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad/k78a-scripts")
from lib import *
A, B = sys.argv[1], sys.argv[2]
ms = int(sys.argv[3]) if len(sys.argv) > 3 else 400
N = int(sys.argv[4]) if len(sys.argv) > 4 else 400
ca, ra = MODS[A]; cb, rb = MODS[B]
ca.VAD_MIN_SILENCE_MS = ms; cb.VAD_MIN_SILENCE_MS = ms
rng = np.random.default_rng(int(sys.argv[5]) if len(sys.argv) > 5 else 0)
bad = {"loud": 0, "vad": 0, "pauses": 0, "plan": 0}
lf_checks = 0
for trial in range(N):
    parts = []
    for _ in range(rng.integers(3, 30)):
        dur = float(rng.choice([0.01, 0.03, 0.1, 0.2, 0.3, 0.45, 0.6, 1.5, 4, 8, 20]))
        lvl = float(rng.uniform(-75, -10))
        parts.append(rng.standard_normal(int(dur * SR)) * 10 ** (lvl / 20))
    wav = np.concatenate(parts).astype(np.float32)
    rms = ca.frame_rms(wav)
    for ratio, rel in [(0.4, 150), (0.6, 150), (0.4, 25), (0.6, 5), (0.4, 50)]:
        lf_checks += 1
        if not np.array_equal(ca.loud_frames(rms, ratio, rel), cb.loud_frames(rms, ratio, rel)):
            bad["loud"] += 1
    if ca._volume_speech_segments(wav) != cb._volume_speech_segments(wav):
        bad["vad"] += 1
    if ra.pauses(wav) != rb.pauses(wav):
        bad["pauses"] += 1
    pa = ca.plan_chunks(wav, target_sec=20.0, max_sec=25.0, context_sec=2.0)
    pb = cb.plan_chunks(wav, target_sec=20.0, max_sec=25.0, context_sec=2.0)
    if tuple(pa) != tuple(pb):
        bad["plan"] += 1
print(f"{A} vs {B} at VAD_MIN_SILENCE_MS={ms}: {N} wavs, {lf_checks} loud_frames checks; mismatches {bad}")
