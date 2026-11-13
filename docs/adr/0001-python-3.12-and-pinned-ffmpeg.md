# ADR 0001 — Python 3.12 and a pinned FFmpeg build

- Status: accepted
- Date: 2026-11-13

## Context

`AGENTS.md` required Python 3.11, but only 3.12.0 and 3.14 exist on the target
machine. The stack also needs FFmpeg with `libass`, which is not installed and
not on `PATH`.

Package metadata constrains the choice:

- `av` 19.x requires Python >= 3.12 (faster-whisper depends on `av`).
- `numpy` 2.5.x ships no cp311 wheel.
- Python 3.14 wheels exist for most packages, but the `faster-whisper` +
  `ctranslate2` + `onnxruntime` combination is not exercised on 3.14.

## Decision

1. Pin **Python 3.12** for the venv (`.venv`), invoked as `py -3.12`. The bare
   `python` command is the Microsoft Store alias and must not be used.
2. Ship **FFmpeg n8.1.3 BtbN win64-gpl static**, downloaded by
   `tools/setup.ps1` into `tools/ffmpeg/bin` (git-ignored) with a pinned URL and
   SHA256. Resolve the binary via `ACF_FFMPEG` -> `tools/ffmpeg/bin` -> `PATH`;
   never assume `PATH`.
3. Verify the binary by capability, not by version string alone: `--enable-libass`,
   filters `ass`/`subtitles`, encoders `libx264`/`aac`, and a real NVENC trial
   encode (`tools/verify_env.py`).

## Consequences

- `AGENTS.md` now states Python 3.12.
- Environment drift is detectable with a single command before any work starts.
- Replacing FFmpeg requires updating the URL, the SHA256 and `docs/env.md`
  together, otherwise `tools/setup.ps1` fails loudly.
- The GPU is treated as optional: NVENC is used for previews, CPU `libx264`
  is the final encoder, and CPU `int8` is the ASR default.
