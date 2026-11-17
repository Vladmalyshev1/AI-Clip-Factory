import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { endpoints } from "@/services/endpoints";
import { LibraryPage } from "@/features/library/LibraryPage";
import type { Clip, Page, Video } from "@/types/api";

vi.mock("@/services/endpoints", () => ({
  endpoints: {
    listLibrary: vi.fn(),
    listVideos: vi.fn(),
    renameClip: vi.fn(),
    deleteClip: vi.fn(),
    retryClip: vi.fn(),
    cancelClip: vi.fn(),
  },
}));

const mocked = vi.mocked(endpoints);

function makeClip(overrides: Partial<Clip> = {}): Clip {
  return {
    id: "clip-1",
    video_id: "video-1",
    kind: "auto",
    parent_clip_id: null,
    status: "completed",
    title: "Insane 1v3",
    start_ms: 10_000,
    end_ms: 40_000,
    duration_ms: 30_000,
    score: 0.92,
    file_size: 1_024,
    width: 1080,
    height: 1920,
    fps: 30,
    aspect_ratio: "9:16",
    output_format: "vertical",
    crop_mode: "pad",
    encoding_mode: "accurate",
    error: null,
    job_id: null,
    created_at_ms: 1_760_000_000_000,
    updated_at_ms: 1_760_000_000_000,
    ...overrides,
  };
}

function renderLibrary() {
  return render(
    <MemoryRouter initialEntries={["/clips"]}>
      <LibraryPage />
    </MemoryRouter>,
  );
}

function emptyVideos(): Page<Video> {
  return { items: [], total: 0, limit: 100, offset: 0 };
}

describe("LibraryPage", () => {
  it("renders cards from real API data with an empty search state", async () => {
    mocked.listVideos.mockResolvedValue({
      items: [{ id: "video-1", title: "VOD" }],
      total: 1,
      limit: 100,
      offset: 0,
    } as unknown as Page<Video>);
    mocked.listLibrary.mockResolvedValue({
      items: [makeClip()],
      total: 1,
      limit: 24,
      offset: 0,
    });

    renderLibrary();

    expect(await screen.findByText("Insane 1v3")).toBeInTheDocument();
    expect(screen.getAllByText("Ready").length).toBeGreaterThan(0);
    expect(screen.getByRole("searchbox")).toBeInTheDocument();
  });

  it("shows the empty state when the backend returns nothing", async () => {
    mocked.listVideos.mockResolvedValue(emptyVideos());
    mocked.listLibrary.mockResolvedValue({ items: [], total: 0, limit: 24, offset: 0 });

    renderLibrary();

    expect(await screen.findByText("No clips yet")).toBeInTheDocument();
  });

  it("shows the API error state and offers a retry", async () => {
    mocked.listVideos.mockRejectedValue(new Error("backend down"));    mocked.listLibrary.mockRejectedValue(new Error("backend down"));

    renderLibrary();

    expect(await screen.findByText(/Could not load the library/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /try again/i })).toBeInTheDocument();
  });

  it("renames a clip through the API", async () => {
    const user = userEvent.setup();
    mocked.listVideos.mockResolvedValue(emptyVideos());
    mocked.listLibrary.mockResolvedValue({
      items: [makeClip()],
      total: 1,
      limit: 24,
      offset: 0,
    });
    mocked.renameClip.mockResolvedValue(makeClip({ title: "Renamed clip" }));

    renderLibrary();
    await user.click(await screen.findByRole("button", { name: "Rename" }));
    const input = screen.getByRole("textbox");
    await user.clear(input);
    await user.type(input, "Renamed clip");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => {
      expect(mocked.renameClip).toHaveBeenCalledWith("clip-1", "Renamed clip");
    });
  });

  it("confirms before deleting a clip", async () => {
    const user = userEvent.setup();
    mocked.listVideos.mockResolvedValue(emptyVideos());
    mocked.listLibrary
      .mockResolvedValueOnce({ items: [makeClip()], total: 1, limit: 24, offset: 0 })
      .mockResolvedValue({ items: [], total: 0, limit: 24, offset: 0 });
    mocked.deleteClip.mockResolvedValue(undefined);

    renderLibrary();
    // The card's delete button (the dialog is not open yet).
    await user.click(await screen.findByRole("button", { name: "Delete" }));

    // Confirmation dialog appears first; nothing deleted yet.
    const dialog = await screen.findByRole("dialog");
    expect(dialog).toBeInTheDocument();
    expect(mocked.deleteClip).not.toHaveBeenCalled();

    await user.click(
      within(dialog).getByRole("button", { name: /delete/i }),
    );
    await waitFor(() => {
      expect(mocked.deleteClip).toHaveBeenCalledWith("clip-1");
    });
  });
});
