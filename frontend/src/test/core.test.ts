/** Unit tests of the shared helpers and API error mapping. */

import { describe, expect, it, vi } from "vitest";
import { ApiError, NetworkError, api } from "@/services/api";
import { errorMessage } from "@/utils/errors";
import { formatDuration, formatScore, isRetryable, isDownloadable } from "@/utils/format";
import { buildLibraryQuery } from "@/features/library/query";

describe("formatDuration", () => {
  it("formats sub-hour durations as m:ss", () => {
    expect(formatDuration(65_000)).toBe("1:05");
    expect(formatDuration(0)).toBe("0:00");
  });

  it("formats hour-long recordings", () => {
    expect(formatDuration(3_723_000)).toBe("1:02:03");
  });

  it("renders a dash for unknown durations", () => {
    expect(formatDuration(null)).toBe("—");
  });
});

describe("formatScore", () => {
  it("keeps two decimals and a dash for null", () => {
    expect(formatScore(0.913)).toBe("0.91");
    expect(formatScore(null)).toBe("—");
  });
});

describe("status helpers", () => {
  it("only failed/canceled clips are retryable", () => {
    expect(isRetryable("failed")).toBe(true);
    expect(isRetryable("canceled")).toBe(true);
    expect(isRetryable("completed")).toBe(false);
    expect(isRetryable("processing")).toBe(false);
  });

  it("only completed clips are downloadable", () => {
    expect(isDownloadable("completed")).toBe(true);
    expect(isDownloadable("failed")).toBe(false);
  });
});

describe("buildLibraryQuery", () => {
  it("maps filters to backend query parameters", () => {
    const search = new URLSearchParams(
      buildLibraryQuery({ videoId: "v1", status: "completed", q: "best", sort: "score", page: 2 }),
    );
    expect(search.get("video_id")).toBe("v1");
    expect(search.get("status")).toBe("completed");
    expect(search.get("q")).toBe("best");
    expect(search.get("sort")).toBe("score");
    expect(search.get("offset")).toBe("48");
    expect(search.get("limit")).toBe("24");
  });

  it("omits empty filters", () => {
    const search = new URLSearchParams(buildLibraryQuery({ page: 0 }));
    expect(search.has("video_id")).toBe(false);
    expect(search.has("q")).toBe(false);
  });
});

describe("errorMessage", () => {
  it("maps ApiError to message + code", () => {
    const message = errorMessage(new ApiError(409, "CLIP_CONFLICT", "already rendering"));
    expect(message).toContain("already rendering");
    expect(message).toContain("CLIP_CONFLICT");
  });

  it("maps network failures to a friendly hint", () => {
    expect(errorMessage(new NetworkError())).toContain("backend");
  });

  it("handles plain strings", () => {
    expect(errorMessage("boom")).toBe("boom");
  });
});

describe("api client", () => {
  it("raises ApiError from the backend error envelope", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: { code: "CLIP_NOT_FOUND", message: "gone" } }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    await expect(api.get("/clips/missing")).rejects.toMatchObject({
      code: "CLIP_NOT_FOUND",
      status: 404,
    });
    vi.unstubAllGlobals();
  });

  it("raises NetworkError when the backend is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("network down")));
    await expect(api.get("/clips")).rejects.toBeInstanceOf(NetworkError);
    vi.unstubAllGlobals();
  });
});
