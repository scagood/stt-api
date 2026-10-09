"""Repro: PARAKEET_VAD_MIN_SILENCE_MS=1000, a quiet speaker's four 0.3 s words 0.6 s apart
(15 dB over -55 dBFS room tone) in a 14 s pause between two 40 s turns at -20 dBFS."""
import lib
lib.set_minsil(1000)
words = [(3.0 + i * 0.9, 0.3, -40) for i in range(4)]
wav, off = lib.pause_between(14.0, words)
print("words at", [(round(off + a, 2), round(off + a + l, 2)) for a, l, _ in words])
for name, mod in (("main", lib.MAIN), ("PR  ", lib.PR)):
    pl = lib.plan(mod, wav)
    print(name, "ranges", lib.secs(pl.ranges))
    print(name, "retime pauses in 40-54 s", [(round(a, 2), round(b, 2)) for a, b in mod[1].pauses(wav) if 40 < b and a < 54])
