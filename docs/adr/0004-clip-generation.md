# ADR 0004 - Clip generation from stored candidates (M6)

- Status: accepted
- Date: 2026-11-17

## Context

M6 turns a stored M4 candidate into a playable file. Everything it needs
already exists: the immutable source recording (M2), the candidate timestamps
(M4) and the FFmpeg layer (M5: `FFmpegRunner`, `build_clip_command`,
`render_clip`, `validate_render`). The open questions are bookkeeping ones:
where the render parameters live, how a clip is retried, cancelled and deleted,
and how a crashed worker resumes.

The machine is a 4 GB GTX 1050 Ti next to an i5-11400F. Re-encoding is the only
reliable path (stream copy snaps to keyframes), so a clip render must be a
single FFmpeg run that can be killed and resumed, and it must never touch the
original recording or leak a half-written file into the UI.

## Decision

1. **One clip row, one job, one stage.** `clips` (migration `0005`) stores the
   resolved window, score, render metadata, error and `job_id`. Generation
   enqueues `Job(type="clip", params={"clip_id": ...})`; the single
   `RenderClipStage` (name `render`) reads the clip row and nothing else.
   `ClipStatus` is `pending/processing/completed/failed/canceled`, with
   `completed` terminal: re-rendering it would overwrite a file the user may
   already have downloaded.
2. **Boundaries are resolved once, by a pure function.** `app/ai/clips.py`
   (`calculate_bounds`) turns a candidate plus `ACF_CLIP_*` limits into
   `[start_ms, end_ms)`: candidate + padding by default, a fixed window around
   the candidate centre on request. Padding is centred under the maximum, the
   window is shifted (never truncated) at the recording edges and grown to the
   minimum. Impossible requests fail with `CLIP_BOUNDS_INVALID` (422) before a
   job exists, so the worker never discovers an invalid spec. The shipped
   maximum equals M4's (`120000` ms), so every detected candidate is
   renderable; `fast` combined with a vertical canvas, a non-9:16 canvas and a
   target duration without `duration_mode="fixed"` are rejected the same way.
3. **Render parameters are snapshotted.** `ClipOptions` is serialised into
   `clips.options_json` at creation time. Changing `ACF_CLIP_*` afterwards
   affects new clips only, and a resumed render uses exactly the settings the
   user asked for. The worker also keeps the resolved `start_ms`/`end_ms`, so a
   re-scored candidate cannot silently move a published clip.
4. **Idempotent generation, conflicts instead of duplicates.** A generate
   request takes SQLite's write lock (`BEGIN IMMEDIATE`), skips candidates that
   already have a pending/processing/completed clip, and re-arms failed or
   canceled ones with a new job. UNIQUE(`highlight_candidate_id`) is the final
   guard: a concurrent duplicate becomes `CLIP_CONFLICT` (409), never two
   renders of the same moment. An empty `candidate_ids` requires `min_score`,
   so a malformed request cannot render a whole VOD by accident.
5. **Every FFmpeg call goes through M5.** `ClipSpec` gained
   `encoding_mode` (`accurate`/`fast`), `output_format` (`vertical`/`original`)
   and `crop_mode` (`pad`/`crop`); the defaults reproduce the previous command
   byte for byte, so M5's contract tests still apply. `fast` stream-copies and
   is therefore limited to `original` without subtitles; `crop` fills the 9:16
   canvas without the blurred background. Cancellation is enforced inside the
   process: the reporter's `check_cancelled` is passed as the runner's
   `cancel_check`, which terminates the child.
6. **Atomic publish.** The render writes `clips/{video_id}/.{clip_id}.part.mp4`,
   which is probed against duration/geometry/audio/aspect-ratio expectations and
   then renamed to `{clip_id}.mp4`. A failure or cancellation deletes the part
   file and records `error_code`/`error_message` on the row. Deleting a clip
   deletes its file, but only when the stored path resolves inside
   `ACF_CLIP_OUTPUT_DIR`, so a corrupted row can never remove unrelated data.
7. **A crashed run resumes from the row.** The stage moves the clip to
   `processing` with the new job id and re-renders from scratch. A user cancel
   (`cancel_requested` on the job) marks the clip `canceled`; a graceful worker
   shutdown returns it to `pending`; a job that exhausts its attempts with a
   lost worker marks the clip `failed` (`WORKER_LOST`) through the queue, so
   `retry` still works. Every bookkeeping write is fenced by the job id: a
   newer job that owns the clip is never overwritten. Completed clips are
   skipped on a second run, even if the source has since been removed.

## Consequences

- Files, not blobs: the database stores only metadata and a path, so the
  application database stays small and backup-friendly. Deleting a video
  cascades its clip rows; the files under `clips/` are left for the user and
  for a future cleanup tool, exactly like other rendered artefacts.
- One worker renders clips sequentially by design; `ACF_CLIP_MAX_CONCURRENT_JOBS`
  only caps how much work may pile up per video.
- `fast` mode is deliberately approximate. It exists for quick iteration on
  long VODs, not for publishing.
- Retry after a *configuration* problem (no disk space, bad mode) works
  because the clip keeps its snapshot; fixing the environment and re-queueing
  is enough.
