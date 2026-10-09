# Review harnesses

Scripts used to review and gate stt-api PRs #65 and #71–#80. They were
written during review, not as maintained tooling. Expect hard-coded paths,
copies of `parakeet_service` modules taken at specific commits, and result
text files next to the scripts that made them. Nothing here is imported by
the service or run by CI.

The bar these scripts checked against: a PR may not cut or drop more speech
than `main`, and may not duplicate or invent words compared with `main`.

## Setup

- Python 3.11+ with `numpy` (and `pytest` for the `test_*.py` probes). The
  real-speech and word-time scripts also need the repo's own dependencies
  (`onnxruntime`, `huggingface_hub`) and `ffmpeg`.
- Most scripts take a tree to test as the first argument (`<worktree>`, or
  `WT=<worktree>`): a checkout of `main` or of the PR head. Main-vs-PR runs
  run the same script with the same seed on both trees and diff the output.
- 126 scripts hard-code the review session's scratch directory. Point them
  at your own copy before running:

  ```sh
  OLD=/tmp/claude-0/-home-user-stt-api/349487e0-d09b-554f-9be8-0221f8cd653b/scratchpad
  grep -rlF "$OLD" review | xargs sed -i "s#$OLD#$PWD/review#g"
  ```

  Paths like `r79a-main`, `k80a-pr` or `r80b-head` in those scripts were
  `git worktree`s of `main` or of the PR head at review time.
- `chunker_*.py`, `*.orig` and `old_*.py` files are snapshots of
  `parakeet_service/chunker.py` (or `retime.py` / `routes.py`) at the
  commit or fix variant in the name. They are loaded by path so that two
  versions can run in one process.

## Data

Synthetic audio is generated in-process from a seed. Only the #80 real-speech
checks need downloads (not committed; about 40 MB of mp3):

- LibriVox, *The Adventures of Sherlock Holmes*, chapters 3 and 5 (public
  domain): `https://archive.org/download/adventures_holmes/adventureholmes_03_doyle_64kb.mp3`
  and `..._05_doyle_64kb.mp3` (listing: `https://archive.org/metadata/adventures_holmes`).
- Convert to 16 kHz mono float32:
  `ffmpeg -nostdin -i holmes_03.mp3 -ac 1 -ar 16000 -f f32le holmes_03.f32`
- Word times: `python k80a-scripts/words_w2v.py holmes_03.f32 words_03.npy`.
  This uses the repo's aligner (`wav2vec2-base-960h` ONNX, fetched from
  Hugging Face on first use): a greedy CTC transcript, then forced alignment
  in groups of about 20 s. `words_03dc.npy` was a second pass over chapter 3
  with the same tool; what it varied was not recorded.
- Parakeet models are not needed. Every harness that "decodes" uses a fake,
  deterministic decoder: each word heard is a function of `(seed, case,
  window)`, so `main` and a PR hear the same words for the same audio.

## Harnesses by PR

Folder prefixes: `rv`/`rr` first review / re-review, `wf` fix checks run in
the review workflows, `v`/`m` merge-readiness checks, `k`/`r`/`c` later
review rounds. The number is the PR.

| Folder | PR | What it checks |
|---|---|---|
| `rv65`, `rr65`, `wf65` | #65 (closed) | Replacing or removing a managed section of a PR body. `bin*/gh` are `gh` stubs; `run.sh <script> <body> [section]` prints the body and edit count. |
| `fuzz/` | #71 | First main-vs-PR chunk fuzz. `fuzz.py <tree> <seed>`; `main.json`/`pr.json` are one run each; `cases.json` the generated cases. |
| `rv71`, `rr71`, `wf71` | #71 | Cuts in pauses after room for context, lead margins. `fuzz.py <tree> <seed> <N>` counts in-fit cuts per file. `repro*.py` hold the findings. |
| `rv72`, `rr72`, `wf72`, `s72/` | #72 | numpy loud/quiet runs, `auto_chunk` removal, the window-grid offset that let a trailing word through (`repro.py`, `test_repro72.py`). `s72/fuzz.py` runs the old and new chunker side by side. |
| `rv73`, `rr73`, `wf73` | #73 | Seam matching in `_stitch`. `invariant.py` checks the documented keep rule; `sim.py <tree> [mishear] [drop] [n]` counts lost and duplicated words at random cuts. |
| `rv74`, `rr74`, `wf74`, `c74`, `c74r` | #74 | Even splits of long speech, slivers, first-range lead margin. `rv74/fz.py <tree> <layout.json> <out>` over `lay*.json`; `out_*.json` are main / base / PR runs. `c74r/fuzz.py <seed>` plus `fix*_s*.txt` / `fuzz_s*.txt` results per seed. |
| `rv75`, `rr75`, `v75`, `m75a`, `m75b`, `wf75` | #75 | Quieter speakers vs breaths and clicks (`loud_frames` relisten), `retime` pauses. `m75b/fuzz.py <seed> <N>` (seed 297 replays as `case297.py`); `manyclicks.py`, `yesbreath.py`, `f2_and_retime.py` are fixed scenes. |
| `r76a`, `r76b`, `speechfirst/` | #76 | Cut at a short pause before the minimum when it costs no extra piece. `fuzz76.py`; `speechfirst/headroom.py` measures how many in-speech cuts were avoidable; `check76.py <N>` compares main with both #76 variants. |
| `k78a`, `k78b`, `v78` | #78 | Quiet sounds joined across 400 ms or `PARAKEET_VAD_MIN_SILENCE_MS`. `k78b/fuzz.py <N> <seed> [min_silence_ms]`; result names encode the run, e.g. `fz_400_0.txt` = min-silence 400, seed 0. `k78b/case.py SEED CASE MINSIL` replays one case. `mutate.sh` checks that the tests catch reverted fixes. |
| `k79a`, `k79b`, `r79a`–`r79d` | #79 (open) | Stalled-piece redo: `_redo_ranges` / `_redo_stretches` and the identity splice. `r79a/fz.py <tree> <one\|two\|three\|batch> <seed> <n> <out.jsonl>`, `cmp.py main.jsonl head.jsonl`; `runall.sh` runs the seed suites whose summaries are `suite-*.txt`. `r79b`/`r79c` `fuzz2.py` track word identity (repeats, short function words, multi-token words). `k79b/edge.py` is the skipped-stretch repro. |
| `k80a`, `k80b`, `r80a`, `r80b` | #80 (open) | Cutting at the quietest gap. `r80a/buckets.py BUCKET N SEED` counts in-word forced cuts, main vs head; `r80b/bounds.py N [scale]` checks piece counts, sizes and windows; `realspeech.py CH` and `margin_real.py` run on the LibriVox data above. |

`misc/` holds loose files from the first review pass (`repro.py` and two
chunker snapshots). `workflows/` holds the Claude Code workflow scripts that
fanned the reviews out to agents. They show the prompts and gates used; they
run only inside Claude Code's Workflow tool.

## Not committed

Audio, `.f32`/`.npy` data, model weights, raw `.jsonl` fuzz dumps (about
11 MB; the `suite-*.txt` / `summ.py` summaries are kept), `.bak` files,
repo snapshots made for fix checks, and any file of 1 MB or more.
