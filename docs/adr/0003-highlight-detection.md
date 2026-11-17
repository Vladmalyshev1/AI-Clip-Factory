# ADR 0003 — Transcript-driven highlight detection (M4)

- Status: accepted
- Date: 2026-11-17

## Context

M4 has to find interesting moments in a 2–5 hour gaming VOD using only what
the machine already has: the completed transcript from M3. No paid APIs, no
external AI services, and no audio-loudness information — faster-whisper
produces text, `avg_logprob` and `no_speech_prob`, nothing else. Whatever
scores a moment must therefore come from the transcript text and its
timestamps, or it must stay out of the score.

## Decision

1. **Pure, deterministic detector behind one function.**
   `app/ai/highlights.py` exposes `detect_highlights(segments, *, duration_ms,
   config)`. It reads no database and writes no files, so a future ML scorer
   can replace the scoring behind the same signature, and the tests run
   without FFmpeg or a model.
2. **Per-segment signals from text only.** Emotion phrases (`EMOTION_PATTERNS`),
   laughter markers, exclamation marks, shouted (upper-case) words, elongated
   vowels and repeated punctuation each contribute a capped weight to a
   segment's `[0, 1]` signal. `no_speech_prob > 0` scales the signal down, so a
   probably-silent window can never become a highlight. **No loudness factor:
   Whisper does not provide one, and inventing it is forbidden.**
3. **Regions, not keyword hits.** Signals are smoothed over a context window
   (`ACF_HIGHLIGHT_CONTEXT_WINDOW_MS`), active segments are grouped into
   regions (gap ≤ `ACF_HIGHLIGHT_MERGE_GAP_MS`), padded on both sides, clamped
   to `[0, duration]` and merged again when the padding makes them overlap.
   A single "wow" can never survive; a sustained burst always can.
4. **Score = weighted blend.** `0.45·intensity + 0.25·support +
   0.15·density + 0.15·duration_fit`, clamped to `[0, 1]`, with the weights as
   module constants. Intensity is the strongest segment, support the mean of
   the smoothed neighbourhood, density the characters-per-second relative to
   the video's own baseline, and duration_fit a trapezoid preference for
   20–90 s moments. `ACF_HIGHLIGHT_MIN_SCORE` and
   `ACF_HIGHLIGHT_MAX_CANDIDATES` trim the result.
5. **One analysis row per video, candidates replaced atomically.** `highlights`
   (UNIQUE `video_id`) plus `highlight_candidates` (migration `0004`). A run
   recomputes the whole candidate list, so a crash or cancel simply returns
   the analysis to `pending`; there is no partial state worth resuming.
   Re-running a completed analysis requires `force=true`, which purges the
   stored candidates — the same contract as M3's `force`.
6. **Job and API mirror M3.** A `highlight` job type runs the single
   `HighlightStage`; `POST /videos/{id}/detect-highlights` (202) refuses a
   missing transcript (404 `TRANSCRIPT_NOT_FOUND`), an incomplete one
   (409 `TRANSCRIPT_INCOMPLETE`), an active job and a completed analysis
   without `force` (409 `HIGHLIGHT_CONFLICT`), all under `BEGIN IMMEDIATE`.
   Reading is `GET /videos/{id}/highlights` plus
   `GET /videos/{id}/highlights/candidates[/{candidate_id}]`.

## Consequences

- Detection is CPU-light (a few thousand segments sort and score in well under
  a second), so the stage checkpoints only at its boundaries; there is no
  per-chunk progress worth persisting.
- Scores are comparable within one video (the density baseline is per video),
  not across videos.
- The rule lists are intentionally small and documented; tuning them is a
  constants change, not a schema change.
