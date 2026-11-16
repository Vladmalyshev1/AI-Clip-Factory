# Environment baseline

Recorded after the M0 bootstrap. Any deviation must be treated as a changed
environment: re-run `tools\verify_env.py` and update this file.

## Verified toolchain

| Component | Version | Notes |
|---|---|---|
| OS | Windows 11 build 26200 | PowerShell 5.1, console code page cp866 |
| Python | 3.12.0 (`.venv`) | `py -3.12`; the `python` alias is the broken Store stub |
| pip | 26.3 | upgraded from the bundled 23.2.1 |
| FFmpeg | n8.1.3-35-gf1acb0a266 | BtbN `win64-gpl-8.1` static build, `tools\ffmpeg\bin` |
| libass / libx264 / freetype / fontconfig | enabled | required for ASS subtitles and final encode |
| NVENC | working | `h264_nvenc` trial encode succeeded; previews only |
| Node.js / npm | v24.11.1 / 11.6.2 | frontend not started yet |
| GPU | GTX 1050 Ti 4 GB, driver 582.53 | Pascal, CC 6.1: no fp16 acceleration, no AV1 |
| CPU / RAM / disk | i5-11400F 6C/12T / 15.9 GB / ~222 GB free | |
| Ollama | 0.40.16, reachable on 127.0.0.1:11434 | model `qwen2.5-coder:7b` (not suitable for 4 GB VRAM) |

## FFmpeg provenance

- URL: `https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-11-13-12-43/ffmpeg-n8.1.3-35-gf1acb0a266-win64-gpl-8.1.zip`
- SHA256: `096d0596ba87ee534c6357797dc68882e714b53aac76859fca4be9c012a6a91c`
- Installed to `tools\ffmpeg\bin` (git-ignored, restored by `tools\setup.ps1`).
- Resolver order used by the app: `ACF_FFMPEG` -> `tools\ffmpeg\bin` -> `PATH`.
  The app must never assume `ffmpeg` is on `PATH`.

## Python packages

Source of truth: `requirements.lock.txt` (frozen from the verified install).
Top-level intent lives in `requirements.in` / `requirements-dev.in`.

Key pins and why they matter:

| Package | Version | Note |
|---|---|---|
| faster-whisper | 1.2.1 | `transcribe()` defaults are `vad_filter=False`, `without_timestamps=False`; always pass every argument explicitly |
| ctranslate2 | 4.9.0 | CPU `int8`; GPU path needs CUDA 12 + cuDNN 9 DLLs (not installed) |
| av | 19.0.2 | requires Python >= 3.12 (this is why 3.12, not 3.11); **incompatible with faster-whisper 1.2.1's `decode_audio`** (see below) |
| onnxruntime | 1.32.0 | Silero VAD comes from faster-whisper; no torch required |
| numpy | 2.5.4 | no cp311 wheel |
| pysubs2 | 1.9.1 | ASS generation/validation |
| pytest | 9.2.0 | markers and timeouts configured in `pyproject.toml` |
| ruff / mypy | 0.17.5 / 2.4.1 | lint and type gates |

## Known environment warnings

- `LongPathsEnabled = 0`. Workspace path is short (27 chars) so this is not a
  failure today; keep generated paths under 240 characters or enable long paths.
- Windows Defender scans multi-GB media writes. Adding an exclusion for
  `var\media` and `.venv` roughly halves render/install time, but needs admin
  rights and is therefore documented, not automated.
- **`av` 19 removed the `metadata_errors` argument** that faster-whisper 1.2.1's
  `decode_audio` still passes, so `WhisperModel.transcribe("<path>")` raises
  `TypeError: open() got an unexpected keyword argument 'metadata_errors'`.
  The ASR layer therefore decodes audio itself (`soundfile`, 16 kHz mono
  `float32`) and always passes a `numpy` array to `transcribe()`.

## Speech recognition (M3)

Local, CPU-only ASR. Settings use the usual `ACF_` prefix.

| Variable | Default | Meaning |
|---|---|---|
| `ACF_ASR_MODEL` | `Systran/faster-whisper-small` | model directory or Hugging Face repo id |
| `ACF_ASR_DEVICE` | `cpu` | `cpu`, `cuda` or `auto` (GPU is never required) |
| `ACF_ASR_COMPUTE_TYPE` | `int8` | verified CPU types: `int8`, `int8_float32`, `int16`, `float32` |
| `ACF_ASR_LANGUAGE` | `en` | forced recognition language |
| `ACF_ASR_CHUNK_SECONDS` | `300` | audio chunk length (30–1800); captured when a transcript starts, see below |
| `ACF_ASR_CPU_THREADS` | `6` | ctranslate2 intra-op threads (6 physical cores) |
| `ACF_ASR_BEAM_SIZE` | `5` | 1–10 |
| `ACF_ASR_VAD_FILTER` | `true` | Silero VAD from faster-whisper (no torch needed) |
| `ACF_ASR_LOCAL_FILES_ONLY` | `true` | never touch the network at runtime |
| `ACF_ASR_CHUNK_MAX_ATTEMPTS` | `3` | retries per chunk before it is reported as failed |
| `ACF_ASR_CONDITION_ON_PREVIOUS_TEXT` | `true` | whisper cross-window conditioning inside a chunk |

Changing `ACF_ASR_CHUNK_SECONDS` affects only transcripts that have not been
split into chunks yet. A resumed transcript keeps the chunk length it was started
with, because its chunk rows are keyed by index; to re-tile a video, re-run it
with `POST /videos/{id}/transcribe` and `force=true`, which purges the chunks.
If the recorded chunks no longer cover the audio, the stage fails with
`TRANSCRIPT_CONFLICT` instead of reporting a partial transcript as complete.

### Model provisioning

The model is loaded lazily from the local Hugging Face cache
(`%USERPROFILE%\.cache\huggingface\hub\models--Systran--faster-whisper-small`,
`model.bin` ≈ 461 MB). It is never downloaded implicitly: a missing model
fails the stage with `ASR_MODEL_UNAVAILABLE` and an explicit instruction.
Fetch it once, on purpose:

```
.\.venv\Scripts\python.exe -m app.ai.model_manager fetch
```

Model files are never committed (the cache lives outside the repository).

### Measured reference (this machine)

`small` / `cpu` / `int8` / 6 threads / `beam_size=5` / VAD on:
model load ≈ 1.1 s, RSS ≈ 435 MB, **RTF ≈ 0.105** (≈9.5× realtime) on a clean
21 s sample. Real VODs are slower; treat any per-hour estimate as provisional
until `tools/asr_benchmark.py` has been run on real footage.

### Benchmarking real footage

The tool mirrors the worker (extract 16 kHz mono → slice chunks → decode) and
prints the realtime factor per chunk. It never modifies the source: audio is
extracted into a temporary directory that is deleted afterwards.

```
.\.venv\Scripts\python.exe tools/asr_benchmark.py "D:\vods\stream.mp4"
.\.venv\Scripts\python.exe tools/asr_benchmark.py stream.mp4 --limit-seconds 600 --json
```

Defaults: `--limit-seconds 600` (first 10 minutes), chunk length from
`ACF_ASR_CHUNK_SECONDS`. Exit code `1` means the source is unreadable, FFmpeg is
missing or the model was never fetched.

### Opt-in smoke test with the real model

The regular suite uses a fake engine. To verify the real model, CTranslate2 and
the audio path on this machine, point the smoke test at any recording with
English speech:

```
$env:ACF_ASR_SMOKE_AUDIO = "D:\vods\stream.mp4"
.\.venv\Scripts\python.exe -m pytest tests/ai/test_asr_smoke.py -m slow -s
```

`ACF_ASR_SMOKE_SECONDS` (default `60`) caps how many seconds are transcribed. The
test is skipped, never failed, when the variable is unset, the file is missing,
FFmpeg is unavailable or the model is not cached.
