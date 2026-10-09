import sys
import numpy as np

root = sys.argv[1]
sys.path.insert(0, root)
from parakeet_service import chunker  # noqa: E402

chunker.VAD = "volume"
chunker.VAD_GATE_DB = None
SR = chunker.TARGET_SR


def db(x):
    return 10 ** (x / 20)


def build(kind, pause_sec, noise_sec, noise_db, speech="noise", seed=0, noise_type="noise"):
    rng = np.random.default_rng(seed)
    total = 40 + pause_sec + 40
    wav = rng.standard_normal(int(total * SR)) * db(-55)
    if speech == "noise":
        wav[: 40 * SR] = rng.standard_normal(40 * SR) * db(-20)
        wav[(40 + pause_sec) * SR:] = rng.standard_normal(40 * SR) * db(-20)
    else:  # tone
        t = np.arange(40 * SR) / SR
        tone = np.sin(2 * np.pi * 220 * t) * np.sqrt(2) * db(-20)
        wav[: 40 * SR] = tone
        wav[(40 + pause_sec) * SR:] = tone
    mid = 40 + pause_sec / 2
    n = int(noise_sec * SR)
    a = int(mid * SR)
    if noise_type == "noise":
        wav[a: a + n] = rng.standard_normal(n) * db(noise_db)
    elif noise_type == "add":
        wav[a: a + n] += rng.standard_normal(n) * db(noise_db)
    return wav.astype(np.float32)


cases = [
    ("a: 10s pause, 10ms click -30", 10, 0.01, -30),
    ("a2: 10s pause, 30ms knock -35", 10, 0.03, -35),
    ("b: 5s pause, 300ms breath -42", 5, 0.3, -42),
    ("ctl: 10s pause, no noise", 10, 0.0, -30),
    ("c: 10s pause, 10ms click -40", 10, 0.01, -40),
    ("c2: 10s pause, 10ms click -35", 10, 0.01, -35),
    ("c3: 10s pause, 20ms click -40", 10, 0.02, -40),
    ("c4: 10s pause, 10ms click -33", 10, 0.01, -33),
    ("c5: 10s pause, 10ms click -34", 10, 0.01, -34),
]
for speech in ("noise", "tone"):
    for name, pause, nsec, ndb in cases:
        wav = build(name, pause, nsec, ndb, speech=speech)
        plan = chunker.plan_chunks(wav, target_sec=60.0, max_sec=75, context_sec=0)
        print(speech, name, [(round(a / SR, 2), round(b / SR, 2)) for a, b in plan.ranges])
