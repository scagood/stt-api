# Changelog

## [2.0.0](https://github.com/scagood/stt-api/compare/v1.5.0...v2.0.0) (2026-10-10)


### ⚠ BREAKING CHANGES

* **service:** long audio is cut at pauses found by loudness instead of by Silero-VAD, so chunk boundaries move and transcripts can differ slightly. Set PARAKEET_VAD=silero for the old behaviour, e.g. for speech over a noise bed within ~5 dB of it, where volume missed some pauses.
* **words:** `align_words` and PARAKEET_ALIGN_WORDS are removed; send `aligner` instead. /health reports aligner state as "name:quantization". A custom catalog's `aligners` entries need the new shape.
* **api:** `model` is required and names a model, not a precision: parakeet-v3-fp32/-fp16/-int8 and the other -quant names are gone (send `quantization` instead, default fp32 also on GPU), as are the aliases (parakeet-tdt-0.6b-v3, HF repo ids, whisper-large, whisper-turbo, ...) and PARAKEET_DEFAULT_MODEL. Nothing is preloaded unless PARAKEET_PRELOAD_MODELS is set. Model cards drop `aliases`, gain `quantizations`, and report `owned_by` as the model's author (nvidia, openai); /health drops `default_model`.
* **models:** PARAKEET_CHUNK_TARGET_SEC and PARAKEET_CHUNK_MAX_SEC are removed; chunk lengths are per model. PARAKEET_CHUNK_MIN_SEC remains.
* Python 3.14 is now the minimum supported interpreter, and the legacy Flask service (`app.py`) and its browser upload page at `/` are removed. The OpenAI-compatible API under `parakeet_service/` is unchanged.

### 🌟 Features

* **api:** name a model and pick its precision separately; fix the Whisper catalog ([da78ff9](https://github.com/scagood/stt-api/commit/da78ff96e0a1cd904e89c8e000fc90b54e63d394))
* **audio:** decode stereo and 24-bit WAVs in process ([#22](https://github.com/scagood/stt-api/issues/22)) ([35d843c](https://github.com/scagood/stt-api/commit/35d843c872fdd01d14fe13a10113067826117d23)), closes [#15](https://github.com/scagood/stt-api/issues/15)
* **compare:** a page to compare models and aligners by ear ([#48](https://github.com/scagood/stt-api/issues/48)) ([1a88458](https://github.com/scagood/stt-api/commit/1a88458247c8f2f1ecb637b433946998c919674d))
* **models:** add opt-in Whisper model support ([#25](https://github.com/scagood/stt-api/issues/25)) ([2e6aaf5](https://github.com/scagood/stt-api/commit/2e6aaf5b78d4a3f56a5c8547e14c56b4a85553b2))
* **models:** opt-in LRU cap on the loaded-model cache ([#37](https://github.com/scagood/stt-api/issues/37)) ([dd71261](https://github.com/scagood/stt-api/commit/dd71261052d1dbeca52bd07a0d2c045cf31457a5))
* **models:** serve parakeet-v3 from Olicorne's re-export ([#42](https://github.com/scagood/stt-api/issues/42)) ([a819f87](https://github.com/scagood/stt-api/commit/a819f8776095879f07d70afefd861e8b56441f7f))
* **models:** YAML model catalog with pinned, explicit files ([#41](https://github.com/scagood/stt-api/issues/41)) ([f2439eb](https://github.com/scagood/stt-api/commit/f2439eb47b38cd256c22597e7107f54616d9fef1))
* require Python 3.14 and drop the legacy Flask service ([#11](https://github.com/scagood/stt-api/issues/11)) ([250dfeb](https://github.com/scagood/stt-api/commit/250dfeb8c55e6401fdb54163915a83b3f9182638))
* **service:** cut speech with no pause in the quietest gap between words ([8837c1d](https://github.com/scagood/stt-api/commit/8837c1d27f0f87ddde8b483b9dbd84a5b3459c0b))
* **service:** find pauses in long audio by volume by default ([b94ca73](https://github.com/scagood/stt-api/commit/b94ca7366a95c02f124479ac429d4eb562abee23))
* **service:** opt-in vad_filter returns nothing for short audio without speech ([#83](https://github.com/scagood/stt-api/issues/83)) ([3e6ac15](https://github.com/scagood/stt-api/commit/3e6ac15d0f64f597b4a5484a3dfdf644c7b0a618))
* **service:** PARAKEET_VAD=volume finds pauses by loudness, ~30x faster ([f6a207b](https://github.com/scagood/stt-api/commit/f6a207bddcf9e3fb1f08acf3e310e71130d99227))
* **service:** unload models and aligners left idle for 6 hours ([#70](https://github.com/scagood/stt-api/issues/70)) ([47e779f](https://github.com/scagood/stt-api/commit/47e779f5508e3f29874e58b5f14e06b333ad79f1))
* **transcripts:** opt-in spoken-form numbers, money and units ([#30](https://github.com/scagood/stt-api/issues/30)) ([d1245e4](https://github.com/scagood/stt-api/commit/d1245e4b35db0286370a724a6182dc846f83fc5d))
* **words:** align English word timestamps with wav2vec2 ([#26](https://github.com/scagood/stt-api/issues/26)) ([f2bc41d](https://github.com/scagood/stt-api/commit/f2bc41d3d8daa291c281a1fe56bd2aaa63c185b3))
* **words:** fp16 aligners; fp16/fp32 aligners on the GPU when the models are ([#57](https://github.com/scagood/stt-api/issues/57)) ([61e699b](https://github.com/scagood/stt-api/commit/61e699b51e26192722a0ff236b256c239e0633fe))
* **words:** make word alignment opt-in per request ([#33](https://github.com/scagood/stt-api/issues/33)) ([9dbf5eb](https://github.com/scagood/stt-api/commit/9dbf5ebb7db157586072661b0870fe585c6981f8))
* **words:** named word aligners from the model catalog, in Parakeet v3's 25 languages ([#45](https://github.com/scagood/stt-api/issues/45)) ([07193b4](https://github.com/scagood/stt-api/commit/07193b43f78d194fbead9aa7c3d6068fb6915fa2))
* **words:** retime_words moves word times out of pauses by loudness ([0464b42](https://github.com/scagood/stt-api/commit/0464b4237f0ca39a7721248d5e528f4009819c41))
* **words:** Whisper word timestamps via forced alignment + real language list ([#34](https://github.com/scagood/stt-api/issues/34)) ([612aec4](https://github.com/scagood/stt-api/commit/612aec4f847f83441fff19c053efb06d6a8ab46e))


### 🩹 Fixes

* **docker:** compose healthchecks wait for /healthz, not /health ([510625e](https://github.com/scagood/stt-api/commit/510625e215b94e31f1c8ed1f9657d09548c4d652))
* **models:** answer 503 naming the model when it cannot be loaded ([#50](https://github.com/scagood/stt-api/issues/50)) ([8c2a679](https://github.com/scagood/stt-api/commit/8c2a679e2c051c8648e02eb7f0dcb2c872d52758))
* **models:** chunk Parakeet v2 at 25/30 s so long audio stops dropping speech ([#39](https://github.com/scagood/stt-api/issues/39)) ([4037c62](https://github.com/scagood/stt-api/commit/4037c62e17457522aaf8ec1c1ec9a857ebb01a09))
* **models:** keep ONNX external data beside its model in the HF cache ([73a44e0](https://github.com/scagood/stt-api/commit/73a44e0e4356b13fa9cf1002c87aa9fa3263e72c)), closes [#35](https://github.com/scagood/stt-api/issues/35)
* **service:** answer aligner_quantization with a 400 naming aligner=name:quantization ([4d06ebc](https://github.com/scagood/stt-api/commit/4d06ebc3b4de3bd64080c44f290c358fd626f61a))
* **service:** call the API stt-api in its docs ([43d3ae8](https://github.com/scagood/stt-api/commit/43d3ae854f4c3fa987b26c3aca7ff53046f2eee8))
* **service:** count a margin frame louder than -50 dBFS as sound, whatever its floor ([d82b41e](https://github.com/scagood/stt-api/commit/d82b41e772b661414a74c530e667afcf60b9f7e6))
* **service:** count every token with a word character when finding a skipped stretch ([4bd641a](https://github.com/scagood/stt-api/commit/4bd641a5cc9c8485981f09d677d9114776648b7c))
* **service:** count only words clear of the first decode's, and lose none, to keep a redo ([174bdf6](https://github.com/scagood/stt-api/commit/174bdf6f4b5b23e12b9b60461726841dcb8e3a78))
* **service:** cut at a short pause before the minimum when it costs no piece ([ae2be04](https://github.com/scagood/stt-api/commit/ae2be04252ace5c98fb9eb96066179c5984f091c))
* **service:** cut parakeet-v2's chunks in pauses again after room for context ([edf9b9e](https://github.com/scagood/stt-api/commit/edf9b9e6b11fa43504b9ffb46dbe35479869876c))
* **service:** document responses, errors and form fields in /docs ([0448965](https://github.com/scagood/stt-api/commit/04489652325c660efab209b14fe63f2b7ae7555a))
* **service:** find a skipped stretch that a lone punctuation token splits ([e26addf](https://github.com/scagood/stt-api/commit/e26addfb16d44c6162c3db695bc801fc603f02b8))
* **service:** hear again what is past a quiet phrase's reach, and keep a sound only half a second past it ([7667b46](https://github.com/scagood/stt-api/commit/7667b46073faaf78e81411217107c39d10125d8f))
* **service:** hear quieter speakers' turns in long audio, not breaths or clicks ([#75](https://github.com/scagood/stt-api/issues/75)) ([936ab1a](https://github.com/scagood/stt-api/commit/936ab1a36bb99a3d1c9cba653f904dfc8567b629))
* **service:** join a quiet stretch's sounds across a fixed 400 ms, not PARAKEET_VAD_MIN_SILENCE_MS ([9ef4e2b](https://github.com/scagood/stt-api/commit/9ef4e2bdc278106b08f1bc7be2519752f4b0a500))
* **service:** join quiet sounds across PARAKEET_VAD_MIN_SILENCE_MS where it is longer than 400 ms ([4c9ed66](https://github.com/scagood/stt-api/commit/4c9ed666a482fcff0e17b24c58596567f0f62d8a))
* **service:** keep a forced cut that already sits in a gap, and score cuts without a DC offset ([553e16b](https://github.com/scagood/stt-api/commit/553e16ba186164066c7c6466807139ff058576ba))
* **service:** keep a quiet word near a phrase whole, not cut where the phrase's reach ends ([9e931e4](https://github.com/scagood/stt-api/commit/9e931e420de5699a1e19db9512498980e0a10ccf))
* **service:** keep a split first or last range's whole margin ([9126951](https://github.com/scagood/stt-api/commit/9126951c259f14623bf4a55afa44938386b48fcf))
* **service:** keep a stalled chunk's redo only if it hears words there ([6841a9f](https://github.com/scagood/stt-api/commit/6841a9ff4f8b2048609210bf442e81734a10b7f2))
* **service:** keep each word at a cut in speech once, not twice or never ([5b63524](https://github.com/scagood/stt-api/commit/5b6352425d6ee3635a914c7d4cfcba7426a338e5))
* **service:** keep silence past the speech from adding a piece ([a3b8220](https://github.com/scagood/stt-api/commit/a3b8220d693b8830b2ccc30f29a7f3b377bcb676))
* **service:** keep sound in a split range's margin, and move a cut only out of a word ([ca8f355](https://github.com/scagood/stt-api/commit/ca8f35501019f424a7c0a4ce42ee853d644ad329))
* **service:** keep the cut before the minimum from cutting speech at other trim gaps ([112759c](https://github.com/scagood/stt-api/commit/112759c943f3282437215295f9b4695dbf0874e2))
* **service:** keep words only one piece heard near a cut by their own time ([308d7d6](https://github.com/scagood/stt-api/commit/308d7d6b1ff6fef17797114c42bf691d3ead4ef3))
* **service:** list timestamp_granularities once in the API docs ([b8bee32](https://github.com/scagood/stt-api/commit/b8bee3235d2d22fcb907c13ded2811e8791c2d41))
* **service:** load fp32 models from an NFS models volume ([#87](https://github.com/scagood/stt-api/issues/87)) ([78f5335](https://github.com/scagood/stt-api/commit/78f53358bf2a92f2261f608300615c798b3e98df))
* **service:** overlap long audio's chunks so Parakeet invents no words at cuts ([#69](https://github.com/scagood/stt-api/issues/69)) ([a9c5036](https://github.com/scagood/stt-api/commit/a9c5036de5c1d06767840580f9841c545ef4b766))
* **service:** redo speech a piece skipped on its own, one-piece clips too ([#79](https://github.com/scagood/stt-api/issues/79)) ([4101302](https://github.com/scagood/stt-api/commit/410130211f2c4cd95a7fba246fdc3f1d0cbde032))
* **service:** size the first range's lead margin by all its phrases ([707aa75](https://github.com/scagood/stt-api/commit/707aa75328f0b37aa0e79c966930c4e66f94c1f3))
* **service:** slide cuts later in the pause and shrink the lead margin to fit ([1bc09b9](https://github.com/scagood/stt-api/commit/1bc09b96ed1f2ae7ee820d61457fd4805d969cdb))
* **service:** split speech with no pause evenly, leaving no sliver ([497b0dc](https://github.com/scagood/stt-api/commit/497b0dcff54279584b743c0f3dfb147a0a82b3f1))
* **service:** square loudness frames in blocks, not the whole file at once ([8b9be2e](https://github.com/scagood/stt-api/commit/8b9be2e27b74e667b87013723d9463bd563cf022))
* **service:** stop calling the -35 dBFS chunks cut at their full length ([0525c14](https://github.com/scagood/stt-api/commit/0525c14e0d577e31579807c809cce90837c0ab44))
* **service:** take a slow DC offset out before measuring loudness ([#82](https://github.com/scagood/stt-api/issues/82)) ([64a9e37](https://github.com/scagood/stt-api/commit/64a9e37e6d3d113b32ddd2fd1e8be242fb43ae34))
* **service:** trim the first phrase's lead margin to the pieces it needs ([45ab84c](https://github.com/scagood/stt-api/commit/45ab84c939cd5faa2334e00d88f33955aa992cff))
* **words:** don't hold the aligner cache lock while downloading a model ([#56](https://github.com/scagood/stt-api/issues/56)) ([6397a85](https://github.com/scagood/stt-api/commit/6397a851dcaee60ccac81244f43a8b1cc69d53fb)), closes [#55](https://github.com/scagood/stt-api/issues/55)


### 📚 Documentation

* add the 1.5.0 to 2.0.0 upgrade guide ([63b735e](https://github.com/scagood/stt-api/commit/63b735e4e34964e8c905ebceb7158ec5ae1b3e66))
* correct the model names, defaults and missing endpoints ([#21](https://github.com/scagood/stt-api/issues/21)) ([c90be47](https://github.com/scagood/stt-api/commit/c90be47db0066449f568bfe89a142a978923fe4e)), closes [#16](https://github.com/scagood/stt-api/issues/16)
* correct what the docs say is true today ([a459ae5](https://github.com/scagood/stt-api/commit/a459ae5440a1f2929f58add8295a75f0fd4a113f))
* describe the gap test and the margin's -50 dBFS rule, and drop figures that held for one recording set ([69c7879](https://github.com/scagood/stt-api/commit/69c7879d7152c7e8f4d1e923d2f3776f3acfe009))
* **docker:** state the GPU image's minimum GPU and driver ([#62](https://github.com/scagood/stt-api/issues/62)) ([5ca261a](https://github.com/scagood/stt-api/commit/5ca261aa1caeed13d1a319944cc88df72f1b64a0))
* restructure the README around using the API; one config reference ([42d6208](https://github.com/scagood/stt-api/commit/42d6208feb8a4d880c68dfc8a5c2f1cb2e341a19))
* say a chunk under the minimum is also cut where that adds no chunk ([ed4be7f](https://github.com/scagood/stt-api/commit/ed4be7f1577493087446d24946a6f02badf60f44))
* say a split range gives up only a shared pause, and what that costs with no context ([07e0a17](https://github.com/scagood/stt-api/commit/07e0a17ab4a99fcb30542c0008c6afa697c71b9c))
* say breaths add up to speech only when closer than 400 ms ([567a233](https://github.com/scagood/stt-api/commit/567a2335ebe940a15cbfa480fe27ba15dd4e8af1))
* **service:** describe how a forced cut moves and what a split range's margin keeps ([d4183b9](https://github.com/scagood/stt-api/commit/d4183b96957f93b8d9462de83881dae46905e794))
* **service:** measure PARAKEET_VAD=volume on three audiobooks and two noise beds ([41c18cb](https://github.com/scagood/stt-api/commit/41c18cbffcaf011d0289c41880fff20b6359415e))
* **service:** say only what was measured about PARAKEET_VAD=volume ([47c00c3](https://github.com/scagood/stt-api/commit/47c00c32fb14ffed0a021c97ec92bc1d7e5b0301))
* state the conditions behind the forced-cut figures, and that v2's gain depends on slack ([1a8a88a](https://github.com/scagood/stt-api/commit/1a8a88ab5e049f43fecf1eee6572a26a551c1a64))
* **words:** Parakeet times words into pauses; name an aligner ([#61](https://github.com/scagood/stt-api/issues/61)) ([900650f](https://github.com/scagood/stt-api/commit/900650fae513e49064e4b448ff9799f82011ca94))


### ⚡ Performance

* **service:** give each audio thread its own Silero VAD ([7400eb2](https://github.com/scagood/stt-api/commit/7400eb20f110be934345ffa8ef62a164c643b6c6))


### 📦 Dependencies

* **pkg:** drop the unused openai and typing_extensions pins ([#18](https://github.com/scagood/stt-api/issues/18)) ([0e25b2c](https://github.com/scagood/stt-api/commit/0e25b2c192a4806bae5d845ab0d20ce3d5e0a3a9)), closes [#12](https://github.com/scagood/stt-api/issues/12)


### 🧹 Chores

* delete the three unreferenced root diagnostic scripts ([#19](https://github.com/scagood/stt-api/issues/19)) ([e2ee3c0](https://github.com/scagood/stt-api/commit/e2ee3c01b98f82bff1345acb9cca0baa28502efb)), closes [#13](https://github.com/scagood/stt-api/issues/13)
* merge upstream groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai@06464cd ([1897584](https://github.com/scagood/stt-api/commit/1897584b8d4ecb47a2e6447150709b8130331fd3))
* merge upstream groxaxo/parakeet-tdt-0.6b-v3-fastapi-openai@06464cd (without TensorRT) ([bcee30b](https://github.com/scagood/stt-api/commit/bcee30b9bea92cfd7dff36331eb9e70c3dbb7083))
* remove the unreferenced parakeet.png ([#20](https://github.com/scagood/stt-api/issues/20)) ([48df6f3](https://github.com/scagood/stt-api/commit/48df6f3b713d487236399309aea9d27c2fb3adab)), closes [#14](https://github.com/scagood/stt-api/issues/14)
* rename the project to stt-api ([422e145](https://github.com/scagood/stt-api/commit/422e1458a1df8a0377b4a74327e76bbb87491cec))


### 🤖 Automation

* **docker:** smoke-test the CPU image by loading parakeet-v3 int8 ([#52](https://github.com/scagood/stt-api/issues/52)) ([ae10c97](https://github.com/scagood/stt-api/commit/ae10c973bfb146e74d96f58f3b206dd358fcd419))

## [1.5.0](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/compare/v1.4.0...v1.5.0) (2026-09-17)


### 🌟 Features

* **service:** size thread pools from cgroup quota, warm up before ready ([#7](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/issues/7)) ([432f728](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/432f728b333c9c53fd0bfc630e049a029aaee005))


### 🩹 Fixes

* **docker:** stop the CPU image pulling CUDA wheels, drop emulated arm64 from PRs ([#10](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/issues/10)) ([db30a89](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/db30a89e7e19266e038ab76e30e4c5ac907ca560))


### 🧹 Chores

* **renovate:** track the Dockerfile.cpu pin and group it with onnxruntime-gpu ([#9](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/issues/9)) ([494e08b](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/494e08b8451b74b570840823318b784ea48b599e))

## [1.4.0](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/compare/v1.3.0...v1.4.0) (2026-08-05)


### 🌟 Features

* **service:** explicit -int8 model ids and aliases in /v1/models ([4360ba9](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/4360ba94f5abbd88bfc525c7e6ed12d727f98725))


### 🩹 Fixes

* **docker:** drop stale PARAKEET_DEFAULT_MODEL pins ([d1aa26f](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/d1aa26fc439abbad9ee0cae1959a5ac1ed8a68c2))

## [1.3.0](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/compare/v1.2.0...v1.3.0) (2026-08-05)


### 🌟 Features

* **service:** OpenAI-compatible /v1/models endpoints ([d609b35](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/d609b350b5ee199b2b85ed09ce36a2f55b2adb17))

## [1.2.0](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/compare/v1.1.0...v1.2.0) (2026-08-05)


### 🌟 Features

* **service:** resolve default model at startup with CUDA probe ([53d2b4b](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/53d2b4bb9d97a8cb94209fa8901c65577f9b9e29))
* **service:** short model names, aliases, and v2/fp16 catalog ([a650eed](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/a650eed247402a9a3b3b9fa63b3d05a205e4bfe5))


### 🩹 Fixes

* sanitize export buttons, add auto-scroll toggle, add /docs endpoint ([02c5b9e](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/02c5b9ea1badbb5ce0f9d6833bfb1ec51424acde))
* **service:** run on Python 3.13+ where audioop is removed ([09759c2](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/09759c22c351db00a4c509313a3401b535919d65))
* **service:** stop word timestamps ballooning across silences ([9cc4ddb](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/9cc4ddbd03f7d583305e539f6e30ff92e17a1578))


### 📚 Documentation

* Add Open WebUI Integration guide ([c01cf71](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/c01cf719350393e3396e445682e995f53e947991))
* Polished install steps and added Parakeet model name support ([8327f66](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/8327f66b2b6ff52d85b5e18253bdc8377506e8d3))
* Update benchmarks to max 30x speedup ([f8d38fd](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/f8d38fd919180659ead08ec91000002abc266519))
* Update README with faster-whisper style comparison ([aebb1a5](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/aebb1a58aa8aced7561fb98b2ad00b7d7f0b72ce))


### 🧹 Chores

* also test 3.13 and 3.14 ([948d441](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/948d4418dd35a3b0d1e81de1c71578a622cfb9cb))


### 🤖 Automation

* add release-please and renovate ([61170bd](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/61170bd0db6c1bb9f91240a31bb1b334810e01a0))
* auto build images ([dd1ddcc](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/dd1ddcc6871ac1b5c628fadfc1c9502f89d7632d))
* install fastapi for route-level unit tests ([ce29273](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/ce2927326d2ff2df57ef3d0bce4d8ffea5e457ea))
* nightly GHCR untagged-image cleanup + editorconfig ([b6639db](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/b6639dbb895901de32b41721cbfaec97ca3dcce5))
* python3.12 test ([73803ce](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/73803ce77566ad0fc9d36eeba0d4e3017f19d85f))
* stub onnx stack in tests when not installed ([a65aaa9](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/a65aaa9f68908d61acfe2bdc35ba988c31b3725b))
* support arm64 too ([f204b71](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/f204b71c274b4c3bc2c9e135549b645c497277a1))
* update to latest action stages ([53b6324](https://github.com/scagood/parakeet-tdt-0.6b-v3-fastapi-openai/commit/53b63248bc4df10f50f85ba3a53eaba8b1df5765))
