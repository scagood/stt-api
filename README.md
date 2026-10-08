# stt-api

[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An OpenAI-compatible speech-to-text server on [ONNX Runtime](https://onnxruntime.ai/).
It serves NVIDIA's [Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
v3 (25 European languages) and v2 (English), and OpenAI's Whisper, on CPU or
GPU. Point an OpenAI client at it and call `/v1/audio/transcriptions`.

Parakeet's Token-and-Duration Transducer (TDT) architecture lets it transcribe
many times faster than real time on a consumer CPU, and faster still on a GPU;
see [Performance](#performance). The project started as a Parakeet v3 server,
which is why its settings are named `PARAKEET_*`.

Upgrading from 1.5.0? Read [UPGRADING.md](UPGRADING.md) first: every request
must now name its model.

## Quick start

### Docker

```bash
docker run -d --name parakeet-cpu -p 5092:5092 -v parakeet-models:/app/models \
    -e PARAKEET_PRELOAD_MODELS=parakeet-v3 ghcr.io/scagood/stt-api:latest-cpu
```

The first start downloads `parakeet-v3` before the server reports ready. For
the GPU image, `docker compose` and the image tags, see [DOCKER.md](DOCKER.md).

### From source

You need Python 3.14 and [FFmpeg](https://ffmpeg.org/). In a virtual
environment (venv or conda):

```bash
git clone https://github.com/scagood/stt-api
cd stt-api
pip install -r requirements.txt
python server.py   # GPU, on :5092
```

`requirements.txt` installs `onnxruntime-gpu`, which has no macOS build. For a
CPU-only install, swap it for `onnxruntime` at the same version, as
`Dockerfile.cpu` does, and turn the GPU off:

```bash
sed 's/^onnxruntime-gpu\[[a-z,]*\]==/onnxruntime==/' requirements.txt > requirements.cpu.txt
pip install -r requirements.cpu.txt
PARAKEET_USE_GPU=false python server.py
```

On Linux, install the CPU build of PyTorch first (`silero-vad` depends on it),
or pip pulls the multi-GB CUDA wheels; `Dockerfile.cpu` shows how.

### First request

```bash
curl http://localhost:5092/v1/audio/transcriptions \
  -F file=@audio.mp3 -F model=parakeet-v3
```

```json
{"text":"The quick brown fox jumps over the lazy dog."}
```

With the OpenAI Python SDK (`pip install openai`; the server doesn't need it):

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:5092/v1", api_key="sk-no-key-required")

with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(model="parakeet-v3", file=f)
print(transcript.text)
```

Swagger UI at [http://localhost:5092/docs](http://localhost:5092/docs) lets
you try every request from the browser.

## Choosing a model

Every request names its `model`; there is no default, and a request without
one is a 422. Add a precision after a colon (`parakeet-v3:fp16`), or send it
as the `quantization` field; if you send both, they must agree. With no
precision you get `fp32`. An unknown model or precision is a 400 that lists
the valid ones.

| Model | Languages | |
|---|---|---|
| `parakeet-v3` | 25 European | The main model |
| `parakeet-v2` | English | |
| `whisper-tiny`, `whisper-base`, `whisper-small`, `whisper-medium`, `whisper-large-v3`, `whisper-large-v3-turbo` | 99 | Word times only [with an aligner](#word-timestamps) |
| `whisper-tiny.en`, `whisper-base.en`, `whisper-small.en`, `whisper-medium.en` | English | Word times only [with an aligner](#word-timestamps) |

Every model comes in `fp32`, `fp16` and `int8`. Parakeet v3 transcribes
Bulgarian, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French,
German, Greek, Hungarian, Italian, Latvian, Lithuanian, Maltese, Polish,
Portuguese, Romanian, Russian, Slovak, Slovenian, Spanish, Swedish and
Ukrainian, without being told which it is hearing.

```bash
curl http://localhost:5092/v1/audio/transcriptions \
  -F file=@audio.mp3 -F model=parakeet-v3:fp16
```

**Choosing a precision.** On a GPU, `fp16` roughly halves VRAM. On a CPU, keep
`fp32`: ONNX Runtime upcasts fp16 there, which is slower. `int8` is the
fastest on CPU, but pick it deliberately: the 1.x int8 export dropped words
after silences and was ~4 WER points worse than fp32 on Spanish. The current
one matched fp32 on English (below) but has no multilingual numbers yet.

**Which export.** `parakeet-v3` runs Olicorne's re-export of NVIDIA's `.nemo`
checkpoint in all three precisions. On a 648 s English audiobook chapter it
scored 1.13% (fp32), 1.07% (fp16) and 1.20% (int8) WER, against 1.20%, 1.20%
and 1.83% for the istupakov/grikdotnet exports it replaced, with int8 ~25%
faster. It was chosen on CPU and has **not been tested on a GPU** or outside
English; the `parakeet-v3` entry in
[`parakeet_service/models.yaml`](parakeet_service/models.yaml) lists what to
revert to if CUDA gives trouble.

**Loading.** Models named in `PARAKEET_PRELOAD_MODELS` load at startup; the
rest load on first request (`PARAKEET_MODEL_CACHE_SIZE` caps how many stay
loaded). A model or aligner, preloaded or not, that no request has used for 6
hours is unloaded to free its memory, and the next request that names it loads
it again, without a warm-up; set `PARAKEET_MODEL_IDLE_TIMEOUT_SEC` to another
number of seconds, or to `-1` to keep everything loaded. A model that can't be
loaded (a failed download, missing from the cache under
`PARAKEET_HF_OFFLINE=true`, refused by ONNX Runtime) answers 503 with a
`detail` naming it and the cause; the next request tries again.

`GET /v1/models` lists every model; `GET /v1/models/parakeet-v3` returns one:

```json
{"id":"parakeet-v3","object":"model","created":1785888000,"owned_by":"nvidia","language":["bg","hr","cs","da","nl","en","et","fi","fr","de","el","hu","it","lv","lt","mt","pl","pt","ro","ru","sk","sl","es","sv","uk"],"quantizations":["fp32","fp16","int8"],"task":"automatic-speech-recognition"}
```

## Response formats

`response_format` is `json` (the default), `text`, `srt`, `vtt` or
`verbose_json`. `verbose_json` always has segments, and has words when you send
`timestamp_granularities[]=word` (`words` is `null` otherwise).
`timestamp_granularities[]` takes `word` and `segment`; anything else is a 400.

```bash
curl http://localhost:5092/v1/audio/transcriptions \
  -F file=@audio.mp3 -F model=parakeet-v3 \
  -F response_format=verbose_json -F 'timestamp_granularities[]=word'
```

```json
{
  "task": "transcribe",
  "language": "auto",
  "duration": 2.3473125,
  "text": "The quick brown fox jumps over the lazy dog.",
  "segments": [
    {"id": 0, "seek": 0, "start": 0.0, "end": 2.3473125,
     "text": "The quick brown fox jumps over the lazy dog.",
     "tokens": [], "temperature": 0.0, "avg_logprob": 0.0,
     "compression_ratio": 0.0, "no_speech_prob": 0.0}
  ],
  "words": [
    {"start": 0.0, "end": 0.16, "word": "The"},
    {"start": 0.16, "end": 0.4, "word": "quick"},
    {"start": 0.4, "end": 0.72, "word": "brown"},
    ...
    {"start": 2.0, "end": 2.3473125, "word": "dog."}
  ]
}
```

A segment's `tokens`, `temperature`, `avg_logprob`, `compression_ratio` and
`no_speech_prob` are always empty or zero; they are there for OpenAI clients
that expect them. `language` echoes the request's, or `auto`.

## Word timestamps

Parakeet's own word times sit on 80 ms frames, and each word's end is
estimated: in the example above, every word ends exactly where the next one
starts. Name an `aligner` and a forced aligner retimes the words from the
audio instead, on 20 ms frames, WhisperX-style but on ONNX Runtime with no
PyTorch. Parakeet still decides the words.

Parakeet's times also slip into pauses: the word before a pause can start
after the speech has stopped, and the word after it before the speech has
started. In a 41-minute LibriVox chapter, 249 words lay entirely inside a
pause (`ffmpeg silencedetect`, -30 dB, 0.3 s or longer), each a real word timed
into the pause beside it. With `aligner=wav2vec2-base-960h`, none did. If you
match words against text or silence, as an audiobook aligner does, name an
aligner, or send `retime_words=true`.

`retime_words=true` finds the pauses by loudness and moves only the words
touching one to its edge: a word that starts in a pause starts at its end, one
that ends in it ends at its start, and one wholly inside goes before it if it
ends a sentence or clause, after it if not. Every other word keeps Parakeet's
time. It costs about 0.1 s per hour of audio, against the aligner's 3.4x the
transcription time on a CPU, but it is not as good: the aligner times every
word from the audio. Against the aligner's times on three LibriVox chapters, it
cut the words lying inside a pause from 249, 68 and 79 to 15, 14 and 43, and
the words starting over 200 ms from the aligner's start from 9.4%, 9.3% and
17.8% to 3.1%, 3.8% and 11.4%.

A pause holds one word at each edge, the last before it and the first after
it. When more than one word wholly inside a pause would go to the same side,
it is no pause: it is speech the gate missed, or times off by more than a
word, and those words keep Parakeet's times rather than pile up at its edge.
A quiet stretch of 3 s or more is heard again at its own level, as when
[finding pauses](#configuration) in long audio, so a quieter speaker's turn
gets pauses of its own instead of being one. In a 25-word turn 20 dB quieter
than the rest, `retime_words` used to move 24 words' starts over 200 ms from
the aligner's (aligning the turn on its own) and squeeze 23 into 40 ms; now 3
and 1, against Parakeet's own 5 and 0. On two LibriVox chapters it moves 8 of
3,782 words differently than before, 7 of them closer to the aligner.

With an aligner named, it only re-times chunks the aligner could not.
`PARAKEET_RETIME_WORDS=true` turns it on for requests that don't say.

```bash
curl http://localhost:5092/v1/audio/transcriptions \
  -F file=@audio.mp3 -F model=parakeet-v3 \
  -F response_format=verbose_json -F 'timestamp_granularities[]=word' \
  -F language=en -F aligner=mms-300m-forced-aligner
```

```python
transcript = client.audio.transcriptions.create(
  model="parakeet-v3",
  file=f,
  response_format="verbose_json",
  timestamp_granularities=["word"],
  language="en",
  extra_body={"aligner": "mms-300m-forced-aligner"},  # non-commercial licence
)
```

The same clip, before and after (seconds):

| Word | Parakeet | `mms-300m-forced-aligner` |
|---|---|---|
| The | 0.00–0.16 | 0.04–0.10 |
| quick | 0.16–0.40 | 0.18–0.36 |
| brown | 0.40–0.72 | 0.42–0.66 |
| fox | 0.72–1.04 | 0.72–0.94 |
| jumps | 1.04–1.36 | 1.12–1.36 |

Name the aligner as you name a model, with its precision after a colon
(`aligner=mms-300m-forced-aligner:fp32`); without one you get `int8`. There is
no default aligner: a request that names none gets Parakeet's times.
`GET /v1/aligners` lists them:

| Aligner | Languages | Error, start / end (English TTS) | License |
|---|---|---|---|
| [`mms-300m-forced-aligner`](https://huggingface.co/onnx-community/mms-300m-1130-forced-aligner-ONNX) | `en` | 37 / 106 ms | **CC-BY-NC-4.0: non-commercial only** |
| [`wav2vec2-large-xlsr-53-english`](https://huggingface.co/Xenova/wav2vec2-large-xlsr-53-english) | `en` | 48 / 104 ms | Apache-2.0 (the model it exports) |
| [`omnilingual-ctc-300m`](https://huggingface.co/OpenVoiceOS/omnilingual-asr-ctc-300m-onnx) | `en` and Parakeet v3's other 24 | 45 / 117 ms | Apache-2.0 |
| [`wav2vec2-base-960h`](https://huggingface.co/onnx-community/wav2vec2-base-960h-ONNX) | `en` | 57 / 131 ms | Apache-2.0 |

For English, use `mms-300m-forced-aligner`: it is the most accurate, and on
real audiobook narration it sounds clearly the best. Its licence is
non-commercial; for commercial use, `wav2vec2-large-xlsr-53-english` is the
next best. `wav2vec2-base-960h` is the smallest and fastest, but the least
accurate. For the other languages, use `omnilingual-ctc-300m`:

```bash
curl http://localhost:5092/v1/audio/transcriptions \
  -F file=@entretien.mp3 -F model=parakeet-v3 \
  -F response_format=verbose_json -F 'timestamp_granularities[]=word' \
  -F language=fr -F aligner=omnilingual-ctc-300m
```

Each aligner comes in `int8` (the default) and `fp32`, which is about as
accurate and 4x the download. All but `omnilingual-ctc-300m` also come in
`fp16`, half the download of `fp32`, for GPUs: on a CPU, keep `fp32` or `int8`,
since ONNX Runtime upcasts `fp16` there, which is slower than `fp32`.

**On a GPU server**, `fp16` and `fp32` aligners run on the GPU, as the models
do. `int8` always runs on the CPU: ONNX Runtime has no CUDA kernels for its
8-bit operations. The default stays `int8` whatever the hardware, so to align
on the GPU, ask for it: `aligner=mms-300m-forced-aligner:fp16`, or
`omnilingual-ctc-300m:fp32`. This has not been tested on a GPU yet;
`PARAKEET_ALIGN_GPU=false` keeps every aligner on the CPU, as before.

Which aligners exist, and the languages each
aligns, is set in the [model catalog](#your-own-model-catalog). An unknown
aligner or precision, or a language the aligner doesn't align, is a 400 that
lists what is available.

**Language.** `language` is a bare ISO 639-1 code (`en`, `fr`), or empty or
`auto`; anything else (`en-US`, `EN`, `English`) is a 400. Nothing detects the
language for alignment: a request without one is aligned as
`PARAKEET_ALIGN_DEFAULT_LANGUAGE`, English unless you change it, so send
`language` for anything else. A chunk mostly in another alphabet (Cyrillic,
Greek, ...) keeps Parakeet's times, but other Latin-script languages would be
aligned as English. English-only models (`parakeet-v2`, `whisper-*.en`) are
aligned as English whatever `language` says.

**Whisper** has no word times of its own, so it returns words only when the
request names an aligner, and a multilingual Whisper model only when the
request also sends `language`. If a chunk can't be aligned at all, its `words`
are `null`, never guessed; a word the aligner can't place sits between its
aligned neighbours.

**Numbers and symbols** are aligned as they are said, by the English
aligners: `42` as "forty two", `$5m` as "five million dollars", `20°C`, `5 May`
as "the fifth of May", `R&D` as "ar and dee". Only the timing uses this; the
text is never changed. A count in year range (`1500`) is heard as a year, so if
it was said "one thousand five hundred" it starts a little late.
`omnilingual-ctc-300m` skips numbers in every language, English included: a
number keeps Parakeet's times, and the words around it are aligned as usual.

**Transcript mistakes.** On clean speech, a word Parakeet missed or got wrong
doesn't drag its neighbours' times; in heavy noise it occasionally still does.
An invented word in a pause takes the pause, but a long one invented in the
middle of continuous speech pushes its neighbours aside.

**Cost.** The aligner runs one request at a time, on its own CPU threads
(`int8`, or any aligner without a GPU) or on the GPU (`fp16` and `fp32` on a
GPU server), so it never holds up other requests' audio decoding. Each
aligner downloads on the first request that names it (int8: ~320 MB, or
~95 MB for `wav2vec2-base-960h`) and adds roughly 3 s per 30 s of audio on a
4-core machine (2 s for `wav2vec2-base-960h`). If a download fails, words keep
Parakeet's times and the load is retried every 5 minutes; `/health` reports
each aligner's state under `aligner`.

## Spoken numbers

Parakeet writes numbers its own way, and not consistently: "twenty-five pounds"
may come back as `£25` or `25 lb`, "ten thirty" as `1030`. With
`spoken_numbers=true`, English transcripts (text, segments, words, every
response format, and the batch endpoint) say numbers, money and units in words,
the way they were said:

| Parakeet wrote | Transcript says |
|---|---|
| `$5`, `cost$25` | five dollars, cost twenty-five dollars |
| `£25`, `25 lb` | twenty-five pounds |
| `$5 million`, `$5m` | five million dollars |
| `20°C`, `50%`, `21st` | twenty degrees Celsius, fifty percent, twenty-first |
| `5 May`, `90 mph` | the fifth of May, ninety miles per hour |
| `MP3`, `COVID-19`, `5m`, `12C` | unchanged: names, or ambiguous |

```bash
curl http://localhost:5092/v1/audio/transcriptions \
  -F file=@audio.mp3 -F model=parakeet-v3 \
  -F spoken_numbers=true -F language=en -F aligner=wav2vec2-base-960h
```

From the OpenAI SDK, send `extra_body={"spoken_numbers": True}`.
`PARAKEET_SPOKEN_NUMBERS=true` turns it on for requests that don't say;
they can still send `spoken_numbers=false`.

**How it chooses.** Different speech often comes out as the same text: `£2.10`
is "two pounds ten" or "two ten", `911` is "nine one one" or "nine eleven". So
the request's [aligner](#word-timestamps) scores each number's readings
against the audio and keeps the one that was said. It also scores likely
mishearings of an amount (`£1.10` for "two pounds ten", `€3` for "thirty
euros"), and replaces Parakeet's number only when the audio clearly prefers
one. Without an aligner, each number gets a fixed reading, usually the most
common one, but a code or a time is read as an amount (`911` as "nine hundred
eleven", `1030` as "one thousand thirty"). The choice was tuned with
`wav2vec2-base-960h`; the other aligners haven't been measured for it yet.

**Cost.** Most chunks with a number need the aligner (97% in our test
corpus): roughly 2 s per 30 s chunk on a 4-core CPU, plus up to 0.9 s to choose
on a dense chunk. They queue with word requests on the aligner's single worker,
so audio with numbers in it is heard at about 13x real time however many
requests are waiting. Requests whose numbers have nothing to decide (`6pm`)
aren't held up.

**Language.** English only, decided as for word timestamps: a request without
`language` is taken as `PARAKEET_ALIGN_DEFAULT_LANGUAGE`. A chunk mostly in
another alphabet is left as written, but Latin-script languages sent without
`language` are rewritten as if English, so send `language`, or set the default
empty if you serve them.

## Batch transcription

`POST /v1/audio/transcriptions/batch` takes several `files` in one request,
with the same `model`, `quantization`, `aligner` and `spoken_numbers` fields
as the single-file endpoint, but no `language` or `response_format`. It
returns text only:

```bash
curl http://localhost:5092/v1/audio/transcriptions/batch \
  -F files=@fox.wav -F files=@fox.aiff -F model=parakeet-v3
```

```json
{"results":[{"filename":"fox.wav","text":"The quick brown fox jumps over the lazy dog.","duration":2.3473125},{"filename":"fox.aiff","text":"The quick brown fox jumps over the lazy dog.","duration":2.3473125}],"batch_size":2}
```

A batch is limited to `PARAKEET_MAX_BATCH_FILES` (16) files and
`PARAKEET_MAX_BATCH_BYTES` (512 MiB); see [request limits](#request-limits).

## Comparing models and aligners by ear

With `PARAKEET_COMPARE_UI=true`, `GET /compare` serves a page for choosing
between them by listening. Pick an audio file and cut it to a clip, then add
rows: a model at a quantization, with or without an aligner (`parakeet-v2:int8`;
`parakeet-v2:int8` + `mms-300m-forced-aligner:int8`; ...). **Compare** sends the
clip through `/v1/audio/transcriptions` once per row and lines up each row's
words under the clip's waveform. Click a word to hear the span that row gave
it; the same word is picked out in every row, and in a table of starts and
ends. The arrow keys step through words and rows, Enter replays, and playback
can be slowed to ½×. A file of exact word times, as synthetic speech has (a
`verbose_json` response or its `words`, or SRT/VTT with one cue per word), adds
a dashed reference row and each row's average error against it.

The browser decodes the file and sends only the clip, as a 16 kHz WAV, so the
page and the server hear the same samples. The page is off by default: each
row is a full transcription, and loads any model or aligner it names, which
then stays loaded until it goes unused for `PARAKEET_MODEL_IDLE_TIMEOUT_SEC`.

## Open WebUI

This server works as [Open WebUI](https://openwebui.com/)'s speech-to-text
engine. Start it (see [Quick start](#quick-start)), then in **Open WebUI
Settings → Audio**:

- **STT Engine**: `OpenAI`
- **OpenAI Base URL**: `http://127.0.0.1:5092/v1`
- **OpenAI API Key**: `sk-no-key-required`
- **STT Model**: `parakeet-v3` (required: the server has no default model)

## Configuration

Every setting is an environment variable, and all are optional. The Docker
images already set the CPU or GPU ones (`PARAKEET_USE_GPU`, `PARAKEET_BATCHED`,
...).

**Server**

| Variable | Default | |
|---|---|---|
| `PARAKEET_HOST` | `0.0.0.0` | bind address |
| `PARAKEET_PORT` | `5092` | bind port |
| `PARAKEET_UVICORN_WORKERS` | `1` | uvicorn worker processes; each loads its own copy of every model |
| `PARAKEET_COMPARE_UI` | `false` | serve the [`/compare` page](#comparing-models-and-aligners-by-ear) |

**Models**

| Variable | Default | |
|---|---|---|
| `PARAKEET_MODELS_DIR` | `models/` in the checkout (`/app/models` in the images) | model cache; must be writable, even when fully seeded |
| `PARAKEET_MODEL_CATALOG` | the built-in `models.yaml` | a catalog file that replaces it; see [below](#your-own-model-catalog) |
| `PARAKEET_PRELOAD_MODELS` | empty | comma-separated `model` (fp32) or `model:quantization` to load and warm up before ready; requests must still name `model` |
| `PARAKEET_MODEL_CACHE_SIZE` | `0` | keep at most N loaded models, and separately N loaded aligners, evicting the least recently used; `0` is no limit |
| `PARAKEET_MODEL_IDLE_TIMEOUT_SEC` | `21600` (6 h) | unload a model or aligner, preloaded or not, that no request has used for this many seconds; the next request that names it loads it again, without a warm-up. `-1` keeps everything loaded |
| `PARAKEET_HF_OFFLINE` | `false` | never contact Hugging Face; every file must already be in the cache |

**Startup**

| Variable | Default | |
|---|---|---|
| `PARAKEET_WARMUP` | `true` | run one synthetic chunk through each preloaded model before ready |
| `PARAKEET_WARMUP_SEC` | `5` | length of that chunk; `0` skips it |
| `PARAKEET_WARMUP_TIMEOUT_SEC` | `120` | a warm-up that fails or takes longer fails startup |

**Hardware and threads**

| Variable | Default | |
|---|---|---|
| `PARAKEET_USE_GPU` | `true` | `true` requires CUDA, `auto` uses it when present, `false` runs on CPU |
| `PARAKEET_GPU_DEVICE_ID` | `0` | CUDA device |
| `PARAKEET_GPU_MEMORY_LIMIT_MB` | `0` | cap on ONNX Runtime's CUDA memory arena, in MiB; `0` is no cap |
| `PARAKEET_GPU_CUDNN_ALGO_SEARCH` | `exhaustive` | how cuDNN picks convolution kernels: `exhaustive`, `heuristic` or `default`; `heuristic` starts faster and uses less memory |
| `PARAKEET_GPU_CUDNN_MAX_WORKSPACE` | `true` | let cuDNN use its largest workspace; `false` saves VRAM, maybe at some speed |
| `PARAKEET_GPU_ARENA_EXTEND_STRATEGY` | `next_power_of_two` | how the CUDA arena grows; `same_as_requested` grows only by what is needed |
| `PARAKEET_BATCHED` | on, unless `PARAKEET_USE_GPU=false` | micro-batch requests together (GPU); off runs parallel single requests (CPU) |
| `PARAKEET_MAX_BATCH_SIZE` | `4` | largest micro-batch |
| `PARAKEET_MAX_BATCH_AUDIO_SECONDS` | `300` | most padded audio in one micro-batch: a batch pads every clip to its longest, so long chunks batch fewer at a time |
| `PARAKEET_BATCH_WINDOW_MS` | `4` | how long to wait to fill a micro-batch |
| `PARAKEET_ORT_INTRA_THREADS` | `1` on GPU; physical cores on CPU | ONNX Runtime threads per inference |
| `PARAKEET_ORT_INTER_THREADS` | `1` | ONNX Runtime inter-op threads |
| `PARAKEET_INFER_WORKERS` | logical CPUs ÷ intra-op threads, at most `4` | parallel inferences on CPU (`PARAKEET_BATCHED` off) |
| `PARAKEET_AUDIO_WORKERS` | physical cores, at most `8` | audio decoding and chunking threads |
| `PARAKEET_ALIGN_THREADS` | physical cores, at most `4` | word aligner threads on the CPU (`int8`, or any aligner without a GPU) |
| `PARAKEET_ALIGN_GPU` | `true` | run `fp16` and `fp32` word aligners on the GPU when the models use it; `false` keeps them all on the CPU |

Core counts respect the affinity mask and the cgroup CPU quota; see
[Running under an orchestrator](#running-under-an-orchestrator).

`/health` reports what each loaded model actually runs on under `runtime`:
`backend` (`cuda` or `cpu`), the providers each ONNX session bound to, and a
`fallback_reason` when CUDA was asked for (`PARAKEET_USE_GPU=auto`) but ONNX
Runtime used the CPU.

**Chunking** (long audio is cut at pauses; each model's chunk length is set in
the catalog)

| Variable | Default | |
|---|---|---|
| `PARAKEET_CHUNK_MIN_SEC` | `20` | shortest chunk before neighbours are merged |
| `PARAKEET_CHUNK_TRIM_SILENCE_SEC` | `3` | cut silences at least this long out of a chunk; the first and last chunks keep up to this much before and after the speech |
| `PARAKEET_CHUNK_CONTEXT_SEC` | `5` | Parakeet only: each chunk also decodes this much of its neighbours' audio either side, from and to a pause, and keeps only the words that start in its own range (see below); at most a quarter of the model's `chunk_max_sec`; `0` turns it off |
| `PARAKEET_VAD` | `volume` | how pauses are found: `volume`, frames quieter than a gate, or `silero`, a speech model, 30x slower or more (see below) |
| `PARAKEET_VAD_GATE_DB` | each file's own | `volume`'s gate in dBFS, such as `-45`; unset, 0.4× the file's average 20 ms frame level (about 8 dB below it), never under -60 dBFS, and a long quiet stretch is heard again at its own level (see below) |
| `PARAKEET_VAD_THRESHOLD` | `0.5` | `silero`'s speech probability |
| `PARAKEET_VAD_MIN_SILENCE_MS` | `400` | shortest pause to cut at |
| `PARAKEET_VAD_SPEECH_PAD_MS` | `120` | padding kept around speech |

**Context at the cuts.** Parakeet makes up a word (`and`, `the`, `I`) when its
input ends shortly after speech. Chunks cut in a pause with no audio past the
cut gained one at about one cut in five. So each chunk now also decodes some
of its neighbours' audio, and keeps only the words that start in its own
range. Both ends of what it decodes go in a pause, because a chunk that starts
or ends inside speech can make Parakeet skip or drop tens of seconds of words:
the nearest pause at least `PARAKEET_CHUNK_CONTEXT_SEC` from the cut, else the
farthest that fits, or, with none, `PARAKEET_CHUNK_CONTEXT_SEC` from the cut.
A chunk that still skips 3 s or more of speech in its own range is decoded
again without context. All of it fits inside the model's `chunk_max_sec`, so a
chunk's own range is shorter by twice the context: `parakeet-v3` still cuts at
about 60 s (at most 65 s), `parakeet-v2` at 20 s instead of 25 s. That decodes
up to 1.25× the audio on `parakeet-v3` and 1.5× on `parakeet-v2`. Long
silences are still cut out, with no context across them. Whisper gets no
context: it returns no word times to trim it back by.

**Finding pauses.** By default a pause is any stretch of 20 ms frames quieter
than the gate, which takes under 0.3 ms of CPU per second of audio.
`PARAKEET_VAD=silero` asks Silero-VAD instead, at about 10 ms per second, all
of it before the first chunk reaches the model: some 10 minutes for a 17-hour
audiobook. Measured on LibriVox narration with `parakeet-v3:int8`, as word
errors against the book's text:

| Audio | `silero` | `volume` | `volume`, gate `-45` |
|---|---|---|---|
| *The Adventures of Sherlock Holmes*, 15 min (2,769 words) | 107 | 102 | 103 |
| *Alice's Adventures in Wonderland*, chapter 3 (1,758) | 144 | 139 | 135 |
| *The War of the Worlds*, chapter 1 (2,292) | 167 | 140 | 143 |
| the Holmes clip under pink noise at -40 dBFS | 107 | 109 | 119 |
| the Holmes clip under pink noise at -35 dBFS | 111 | 116 | 116 |

Silero's run of *The War of the Worlds* lost a sentence that volume's kept;
where chunks fall decides that, so take it as luck rather than an advantage.
Volume needs pauses quieter than the speech. Under a noise bed the file's own
gate rises with the noise: at -40 dBFS it still found every pause, at -35 dBFS
(5 dB under the speech) it missed some and 10 chunks were cut mid-speech at
their full length. A fixed gate below the noise finds no pause at all, and
every chunk is cut that way. Music hasn't been measured.

**Quieter speakers.** The file's own gate follows its average, so a speaker
far quieter than the rest (a remote guest, a phone leg, a question from the
audience) can sit under it for a whole turn, which would then be cut out as a
long silence and never decoded. So a quiet stretch at least
`PARAKEET_CHUNK_TRIM_SILENCE_SEC` long is heard again at its own level: 100 ms
of it louder than 0.4× its own average, and 10 dB over its quietest tenth, is
speech. Room tone stays within a few dB of its floor, so a long pause stays
one. With a 15 s turn (25 words) of one LibriVox reader between two minutes of
another:

| Quiet turn | words heard before | now |
|---|---|---|
| 14 dB quieter | 25 | 25 |
| 20 dB quieter | 6, and an "Oh." not said | 25 |
| 26 dB quieter | 0 | 25 |

On other recordings of the three chapters above (not the table's), and the
two noise beds, the transcripts were the same but for one word: Holmes' first
8 s of room noise rise 14 dB over its floor for 100 ms at a time, so they are
decoded now, which moved where Parakeet put a word at the first cut and both
chunks kept it (90 word errors, now 91). A quiet stretch that never rises
10 dB over its own floor, such as a steady tone or speech under noise within
10 dB of it, is still a pause. A fixed `PARAKEET_VAD_GATE_DB` is never heard
again: everything under it is silence.

**Words and numbers**

| Variable | Default | |
|---|---|---|
| `PARAKEET_ALIGN_DEFAULT_LANGUAGE` | `en` | language assumed for [word timestamps](#word-timestamps) and [spoken numbers](#spoken-numbers) when a request sends none; empty uses them only when `language` is sent |
| `PARAKEET_SPOKEN_NUMBERS` | `false` | spoken numbers for requests that don't send `spoken_numbers` |
| `PARAKEET_RETIME_WORDS` | `false` | [`retime_words`](#word-timestamps) for requests that don't send it: move Parakeet's word times out of pauses |

### Request limits

A request over a limit is rejected with 413.

| Variable | Default | |
|---|---|---|
| `PARAKEET_MAX_UPLOAD_BYTES` | 256 MiB | one uploaded file |
| `PARAKEET_MAX_AUDIO_SECONDS` | `7200` (2 h) | one file's decoded duration |
| `PARAKEET_MAX_REQUEST_CHUNKS` | `512` | chunks one request may produce |
| `PARAKEET_MAX_BATCH_FILES` | `16` | files in one batch request |
| `PARAKEET_MAX_BATCH_BYTES` | 512 MiB | total bytes in one batch request |
| `PARAKEET_FFMPEG_TIMEOUT_SEC` | `180` | per-request FFmpeg decode time (not a 413) |

### Your own model catalog

The models and aligners are defined in
[`parakeet_service/models.yaml`](parakeet_service/models.yaml). Each model has
its family, languages and chunk lengths, and per precision a Hugging Face repo,
a pinned commit and the files that differ from the defaults:

```yaml
models:
  parakeet-v2:
    family: parakeet
    onnx_asr_type: nemo-conformer-tdt
    languages: *english            # from the file's `languages` section
    chunk_target_sec: 25.0
    chunk_max_sec: 30.0
    quantizations:
      int8:
        repo: istupakov/parakeet-tdt-0.6b-v2-onnx
        revision: "0bbb45a3365852604aef28b538a8f066f4ccaa85"
        files:
          encoder-model.onnx: encoder-model.int8.onnx
          decoder_joint-model.onnx: decoder_joint-model.int8.onnx
```

The `aligners` section lists the [word aligners](#word-timestamps) the same
way, plus the languages each aligns and how to spell a transcript for it
(`aligners: {}` serves none). A quantization marked `cpu_only: true` stays on
the CPU on a GPU server; the built-in catalog marks the `int8` ones. To serve a different set without rebuilding the
image, point `PARAKEET_MODEL_CATALOG` at another file of the same shape. It
**replaces** the built-in catalog, so copy the built-in file and edit it. The
service checks it at startup and refuses to start on a mistake, naming it; it
is read only then, so restart after changing it.

On Kubernetes, keep it in a ConfigMap:

```bash
kubectl create configmap parakeet-models --from-file=models.yaml=parakeet_service/models.yaml
```

```yaml
# in the Deployment's pod spec
containers:
  - name: parakeet
    env:
      - name: PARAKEET_MODEL_CATALOG
        value: /config/models.yaml
    volumeMounts:
      - name: model-catalog
        mountPath: /config
volumes:
  - name: model-catalog
    configMap:
      name: parakeet-models
```

### Running under an orchestrator

**CPU limits are quotas, not cpusets.** A Kubernetes `resources.limits.cpu` is
invisible to `sched_getaffinity()` and `psutil`, which report the node's full
core count, so without care a 4-core pod on a 64-core node would start 64 ONNX
Runtime threads. Thread pools are sized from the cgroup quota instead, read
from the process's own cgroup and its ancestors, so it is found under systemd
`CPUQuota=` and `--cgroupns=host` as well. `/health` reports `cgroup_quota`
next to the detected core counts under `cpu`; set `PARAKEET_ORT_INTRA_THREADS`
to override.

**Cold start.** Nothing loads at startup unless it is listed in
`PARAKEET_PRELOAD_MODELS`. Each listed model is then warmed up before
`/healthz` answers 200, so the first real request doesn't pay ONNX Runtime's
setup. A warm-up that fails or exceeds `PARAKEET_WARMUP_TIMEOUT_SEC` fails
startup rather than reporting a replica ready that can't run inference. A
preloaded model unloaded after `PARAKEET_MODEL_IDLE_TIMEOUT_SEC` unused is
loaded again by the next request that names it, which pays for the load and the
setup the warm-up hid; `/healthz` stays 200 meanwhile. The
healthcheck `start_period` in the Dockerfiles and `docker-compose.yml` allows
for model load plus that timeout; raise both together. With a pre-seeded cache,
set `PARAKEET_HF_OFFLINE=true` to skip the Hugging Face revision check each
start makes.

## Performance

| Workload | CPU: i7-12700KF, int8 | GPU: RTX 3090, fp32 |
|---|---|---|
| One 300 s file | 10.41 s (27.2× real time) | 1.37 s (205.9×) |
| 16 × 10 s files at once | 39.3× throughput | 200.3× throughput |

Measured in 1.x, with the `parakeet-v3` exports 2.0 replaced. The speed comes
from cutting long audio at pauses with Silero-VAD and running the chunks in
parallel (micro-batched on GPU), decoding each upload once (16 kHz PCM WAV
without FFmpeg at all), and sizing every thread pool to the CPUs actually
available. [OPTIMIZATION.md](OPTIMIZATION.md) has the method, what didn't
work, the GPU sweeps, and accuracy benchmarks against Whisper.

## Acknowledgments

This project stands on the shoulders of giants and wouldn't be possible without:

- **[Shadowfita](https://github.com/Shadowfita/parakeet-tdt-0.6b-v2-fastapi)** - For the original FastAPI implementation that served as the foundation for this project
- **[NVIDIA](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)** - For developing and open-sourcing the exceptional Parakeet TDT model family
- **[groxaxo](https://github.com/groxaxo)** - The mastermind behind this project, bringing together ONNX optimization, multilingual support, and seamless OpenAI API compatibility

Thank you to all contributors and the open-source community for making high-performance, local speech recognition accessible to everyone!
