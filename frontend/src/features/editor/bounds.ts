/** Clip window arithmetic used by the editor timeline (mirrors backend rules). */

export const MIN_DURATION_MS = 1_000;
export const MAX_DURATION_MS = 600_000;

interface BoundsInput {
  startMs: number;
  endMs: number;
  videoMs: number;
}

/** Normalise a draft window against the recording and duration limits. */
export function clampBounds({ startMs, endMs, videoMs }: BoundsInput): {
  startMs: number;
  endMs: number;
} {
  let start = Math.max(0, startMs);
  let end = Math.min(Math.max(endMs, start + MIN_DURATION_MS), videoMs);
  if (end - start < MIN_DURATION_MS) {
    // Too short at the recording's end: keep the end, shift the start left.
    start = Math.max(0, end - MIN_DURATION_MS);
  }
  // Enforce the maximum by trimming from the far side.
  if (end - start > MAX_DURATION_MS) {
    end = start + MAX_DURATION_MS;
  }
  return { startMs: Math.round(start), endMs: Math.round(end) };
}

export function durationError(durationMs: number): string | null {
  if (durationMs < MIN_DURATION_MS) return `Clip must be at least ${MIN_DURATION_MS / 1000} s long`;
  if (durationMs > MAX_DURATION_MS) return `Clip must be at most ${MAX_DURATION_MS / 1000} s long`;
  return null;
}
