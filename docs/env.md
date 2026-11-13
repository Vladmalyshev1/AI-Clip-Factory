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
| faster-whisper | 1.2.1 | defaults changed: `vad_filter=True`, `without_timestamps=True`; pass all args explicitly |
| ctranslate2 | 4.9.0 | CPU `int8`; GPU path needs CUDA 12 + cuDNN 9 DLLs (not installed) |
| av | 19.0.2 | requires Python >= 3.12 (this is why 3.12, not 3.11) |
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
