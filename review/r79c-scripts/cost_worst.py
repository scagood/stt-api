import logging, sys
sys.path.insert(0, sys.argv[1]); sys.path.insert(1, "/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/k79a-scripts".replace("/k79a", "/scratchpad/k79a"))
from simlib import Model, run, words_every
logging.disable(logging.CRITICAL)
truth = words_every(0.3, 120, 0.4)
# every decode hears one word in 3.5 s (music with sparse lyrics); VAD all speech
m = Model(truth, skips=lambda a, b: [(x * 3.5 + 0.5, x * 3.5 + 3.5) for x in range(40)])
out = run(120, [(0, 60), (60, 120)], [(0, 65), (55, 120)], [(0, 120)], m)
redo = out.calls[2:]
print(f"redo decodes={len(redo)} seconds={sum(b - a for a, b in redo):.1f} calls={redo} words={len(out.words)}")
