# ADR 0002 — Chunked, resumable speech recognition on CPU

- Status: accepted
- Date: 2026-11-17

## Context

M3 has to transcribe 2–5 hour recordings on an i5-11400F / 16 GB / GTX 1050 Ti
machine, without paid APIs, without loading a whole recording into RAM, and
resumably: a crash, a cancel or a graceful worker shutdown must not throw away
hours of work. faster-whisper 1.2.1 with `Systran/faster-whisper-small` on CPU
`int8` measures RTF ≈ 0.105, so a 4 hour VOD is ≈ 25 minutes of CPU time — long
enough that losing it to a single crash is unacceptable.

Two environment facts shaped the design:

1. `faster_whisper.audio.decode_audio` calls `av.open(..., metadata_errors=...)`,
   which PyAV 19.0.2 rejects (`TypeError: unexpected keyword argument`). Passing
   a *path* or file object to `transcribe()` therefore fails on this machine.
2. AAC/MP3 padding makes an extracted WAV slightly longer than the probed video
   duration (≈ 22 ms on a 61 s sample), and `-c copy` slicing adds ≈ 8 ms to
   every chunk tail.

## Decision

1. **Decode audio ourselves and pass a NumPy array.** `app/media/audio.py`
   extracts a mono 16 kHz `pcm_s16le` WAV with FFmpeg (streamed to disk, never
   read whole), `app/media/audio_chunks.py` slices fixed windows, and
   `soundfile` decodes each *chunk* to a 1-D `float32` array. Only one short
   chunk (≈ 19 MB for 5 minutes) is ever in RAM.
2. **Chunk at 300 s, re-encode every slice.** `-ss`/`-t` are input options and the
   chunk is re-encoded to PCM instead of stream-copied: measured output is
   exactly 3.000 s with identical samples, while `-c copy` adds ≈ 8 ms of tail.
3. **Absolute timeline is `offset_ms + round(segment.start * 1000)`.** With
   `vad_filter=True` faster-whisper restores the original timeline, so absolute
   timestamps hold for every segment. `avg_logprob` and `no_speech_prob` are
   stored as-is: they belong to a 30 s decode window, not to a segment.
4. **Resume at chunk granularity, commit per chunk.** `transcripts`,
   `transcript_chunks` and `transcript_segments` (migration `0003`) store one
   transcript per video, one row per 300 s window and the recognised segments.
   Each chunk result is written in its own transaction and replaces previous rows
   (`delete` + `insert`), so re-running a chunk is idempotent. A `running` chunk
   left by a dead worker is claimable again — which is why
   `TranscriptChunkStatus.RUNNING → RUNNING` is an allowed transition ("claimed
   again"), mirroring the existing `running → running` tolerance for transcripts.
5. **The stage cannot complete a transcript with unfinished chunks.**
   `complete_transcript` refuses and reports the unfinished indexes; a failure
   marks the chunk `failed` (still claimable) and the transcript `failed`, while a
   cancellation returns the transcript to `pending` and keeps every finished
   chunk.
6. **Pipelines are selected by `Job.type`.** A single global stage registry could
   not host both the queue-semantics `noop`/`noop2` pipeline and ASR without
   breaking M1/M2, so `PIPELINES`/`stages_for(job.type)` replaced `REGISTRY`;
   `"pipeline"` is unchanged and unknown types fall back to it.
7. **`StageContext` gained optional `settings` and `session_factory`.** A stage
   had no other way to reach configuration or a short-lived session; no open
   session is ever passed around, so long media operations never hold a SQLite
   connection.
8. **One model per process, loaded lazily, guarded across processes.**
   `AsrModelManager` resolves the model from the local Hugging Face cache
   (`local_files_only=True` by default) and loads it on first use;
   `AsrProcessLock` (atomic `mkdir` + pid file with stale-lock breaking) prevents
   two workers from allocating the model at once. A missing model is
   `ASR_MODEL_UNAVAILABLE` (503) with the exact fetch command, never a crash.
   Downloading happens only through `python -m app.ai.model_manager fetch`.
9. **Force means purge.** `POST /videos/{id}/transcribe` reuses an interrupted
   transcript, refuses a completed one unless `force=true`, and refuses a video
   with a live transcription job (one job per video at a time).
10. **The chunk grid is part of the transcript, not a live setting.**
    `transcripts.chunk_seconds` is written once, when the transcript has no chunk
    rows (fresh or purged by `force`), and every later run re-plans with the
    recorded value (`transcript_service.adopt_chunk_grid`). Chunk rows are keyed
    by `chunk_index`, so re-tiling a partially finished transcript would either
    duplicate speech (new smaller chunks inside a finished one) or silently drop
    it (one bigger chunk swallowing several unfinished ones). A completed chunk
    row whose window does not match the plan is a `TRANSCRIPT_CONFLICT` (409) and
    asks for `force=true`, and `complete_transcript` additionally refuses a row
    set that does not tile `[0, expected_ms)`.
11. **Cancellation is polled, not inferred from progress.**
    `FFmpegRunner.run(cancel_check=...)` invokes the job's cancellation callback
    on every loop iteration, so extraction and slicing stay cancellable even when
    FFmpeg stops emitting progress lines entirely.

## Consequences

- A crash costs at most one 5 minute chunk plus its FFmpeg slicing.
- Extraction is repeated only when the existing `audio.wav` does not match the
  expected duration/format; a successful run deletes it.
- The API can page segments with millisecond timestamps and report chunk-level
  progress without decoding anything.
- Chunk boundaries are derived from the *extracted* audio duration, so a tail
  chunk can be a few milliseconds longer than `video.duration_ms` and a segment
  may end just past it; consumers clamp when they render.
- `condition_on_previous_text` stays enabled inside a chunk but is never shared
  across chunks, so the 300 s boundaries are natural context resets.
- Changing `ACF_ASR_CHUNK_SECONDS` does **not** re-tile a transcript that already
  has chunks: the run logs the mismatch and keeps the recorded grid. To transcribe
  a video with a different chunk length, re-run it with `force=true`, which purges
  the chunks and adopts the new value.
