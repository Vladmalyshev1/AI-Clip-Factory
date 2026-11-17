/** Bounds validation shared by the editor UI (mirrors backend rules). */

import { describe, expect, it } from "vitest";
import { clampBounds, durationError } from "@/features/editor/bounds";

describe("clampBounds", () => {
  it("keeps the start inside the recording", () => {
    expect(clampBounds({ startMs: -500, endMs: 5_000, videoMs: 60_000 })).toEqual({
      startMs: 0,
      endMs: 5_000,
    });
  });

  it("clamps the end to the recording length", () => {
    expect(clampBounds({ startMs: 55_000, endMs: 90_000, videoMs: 60_000 })).toEqual({
      startMs: 55_000,
      endMs: 60_000,
    });
  });

  it("enforces the minimum duration by moving the end right", () => {
    expect(clampBounds({ startMs: 10_000, endMs: 10_500, videoMs: 60_000 })).toEqual({
      startMs: 10_000,
      endMs: 11_000,
    });
  });

  it("prefers moving the start left over shortening at the video end", () => {
    expect(clampBounds({ startMs: 59_200, endMs: 60_000, videoMs: 60_000 })).toEqual({
      startMs: 59_000,
      endMs: 60_000,
    });
  });

  it("enforces the maximum duration", () => {
    expect(clampBounds({ startMs: 0, endMs: 700_000, videoMs: 900_000 })).toEqual({
      startMs: 0,
      endMs: 600_000,
    });
  });
});

describe("durationError", () => {
  it("accepts a healthy window", () => {
    expect(durationError(30_000)).toBeNull();
  });

  it("rejects inverted and oversized windows", () => {
    expect(durationError(0)).toContain("at least");
    expect(durationError(700_000)).toContain("at most");
  });
});
