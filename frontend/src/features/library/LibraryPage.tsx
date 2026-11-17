import { useCallback, useEffect, useState } from "react";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { Spinner, StatePanel } from "@/components/StatePanel";
import { endpoints } from "@/services/endpoints";
import { toast } from "@/hooks/useToasts";
import { errorMessage } from "@/utils/errors";
import { isDownloadable } from "@/utils/format";
import { ClipCard } from "./ClipCard";
import { useLibrary } from "./useLibrary";
import type { Clip } from "@/types/api";
import { useSearchParams } from "react-router-dom";

const STATUS_OPTIONS = [
  { value: "", label: "All statuses" },
  { value: "completed", label: "Ready" },
  { value: "processing", label: "Processing" },
  { value: "pending", label: "Pending" },
  { value: "failed", label: "Failed" },
  { value: "canceled", label: "Canceled" },
];

export function LibraryPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const videoFilter = searchParams.get("video") ?? "";
  const statusFilter = searchParams.get("status") ?? "";
  const kindFilter = searchParams.get("kind") ?? "";
  const sort = (searchParams.get("sort") ?? "date") as "date" | "score";
  const search = searchParams.get("q") ?? "";
  const page = Math.max(0, Number(searchParams.get("page") ?? "0") || 0);

  const [searchDraft, setSearchDraft] = useState(search);
  const [reloadToken, setReloadToken] = useState(0);
  const [videos, setVideos] = useState<Array<{ id: string; title: string }>>([]);
  const [busy, setBusy] = useState(false);
  const [renaming, setRenaming] = useState<Clip | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [deleting, setDeleting] = useState<Clip | null>(null);

  const pageSize = 24;
  const library = useLibrary(
    {
      videoId: videoFilter || undefined,
      status: statusFilter || undefined,
      kind: (kindFilter || undefined) as "auto" | "export" | undefined,
      q: search || undefined,
      sort,
      page,
      pageSize,
    },
    reloadToken,
  );

  useEffect(() => {
    endpoints
      .listVideos(100)
      .then((result) => setVideos(result.items.map((v) => ({ id: v.id, title: v.title }))))
      .catch(() => setVideos([]));
  }, []);

  const updateParams = useCallback(
    (patch: Record<string, string | null>) => {
      const next = new URLSearchParams(searchParams);
      for (const [key, value] of Object.entries(patch)) {
        if (value === null || value === "") next.delete(key);
        else next.set(key, value);
      }
      if (!("page" in patch)) next.delete("page");
      setSearchParams(next);
    },
    [searchParams, setSearchParams],
  );

  const withBusy = async (action: () => Promise<void>): Promise<void> => {
    setBusy(true);
    try {
      await action();
      setReloadToken((token) => token + 1);
    } catch (cause) {
      toast.error(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  };

  const totalPages = Math.max(1, Math.ceil(library.total / pageSize));

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-lg font-semibold text-slate-100">Clip Library</h1>
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="search"
            value={searchDraft}
            onChange={(event) => setSearchDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") updateParams({ q: searchDraft });
            }}
            onBlur={() => updateParams({ q: searchDraft })}
            placeholder="Search by title…"
            className="w-44 rounded-lg border border-surface-border bg-surface px-3 py-1.5 text-sm placeholder:text-slate-500 focus:border-accent focus:outline-none"
          />
          <select
            value={videoFilter}
            onChange={(event) => updateParams({ video: event.target.value })}
            className="max-w-44 rounded-lg border border-surface-border bg-surface px-2 py-1.5 text-sm focus:border-accent focus:outline-none"
            aria-label="Filter by video"
          >
            <option value="">All videos</option>
            {videos.map((video) => (
              <option key={video.id} value={video.id}>
                {video.title}
              </option>
            ))}
          </select>
          <select
            value={statusFilter}
            onChange={(event) => updateParams({ status: event.target.value })}
            className="rounded-lg border border-surface-border bg-surface px-2 py-1.5 text-sm focus:border-accent focus:outline-none"
            aria-label="Filter by status"
          >
            {STATUS_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <select
            value={kindFilter}
            onChange={(event) => updateParams({ kind: event.target.value })}
            className="rounded-lg border border-surface-border bg-surface px-2 py-1.5 text-sm focus:border-accent focus:outline-none"
            aria-label="Filter by type"
          >
            <option value="">All types</option>
            <option value="auto">Auto</option>
            <option value="export">Exports</option>
          </select>
          <select
            value={sort}
            onChange={(event) => updateParams({ sort: event.target.value })}
            className="rounded-lg border border-surface-border bg-surface px-2 py-1.5 text-sm focus:border-accent focus:outline-none"
            aria-label="Sort"
          >
            <option value="date">Newest first</option>
            <option value="score">Best score</option>
          </select>
        </div>
      </div>

      {library.loading && library.clips.length === 0 ? (
        <Spinner />
      ) : library.error !== null ? (
        <StatePanel
          title="Could not load the library"
          hint={library.error}
          action={
            <button
              type="button"
              onClick={library.refresh}
              className="rounded-lg bg-accent px-3 py-1.5 text-sm text-white hover:bg-accent-soft"
            >
              Try again
            </button>
          }
        />
      ) : library.clips.length === 0 ? (
        <StatePanel
          title="No clips yet"
          hint="Generate clips from a video's highlight candidates, then find them here."
        />
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {library.clips.map((clip) => (
              <ClipCard
                key={clip.id}
                clip={clip}
                busy={busy}
                onRename={(target) => {
                  setRenaming(target);
                  setRenameDraft(target.title ?? "");
                }}
                onDelete={setDeleting}
                onRetry={(target) =>
                  void withBusy(async () => {
                    if (isDownloadable(target.status)) return;
                    if (target.status === "pending" || target.status === "processing") {
                      await endpoints.cancelClip(target.id);
                      toast.info("Render cancelled");
                    } else {
                      await endpoints.retryClip(target.id);
                      toast.success("Re-queued for rendering");
                    }
                  })
                }
              />
            ))}
          </div>
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-2 text-sm text-slate-400">
              <button
                type="button"
                disabled={page === 0}
                onClick={() => updateParams({ page: String(page - 1) })}
                className="rounded-lg px-3 py-1.5 hover:bg-slate-700/40 disabled:opacity-30"
              >
                ← Prev
              </button>
              <span>
                Page {page + 1} of {totalPages} · {library.total} clips
              </span>
              <button
                type="button"
                disabled={page + 1 >= totalPages}
                onClick={() => updateParams({ page: String(page + 1) })}
                className="rounded-lg px-3 py-1.5 hover:bg-slate-700/40 disabled:opacity-30"
              >
                Next →
              </button>
            </div>
          )}
        </>
      )}

      <ConfirmDialog
        open={deleting !== null}
        title="Delete this clip?"
        body={`"${deleting?.title ?? "clip"}" and its rendered file will be removed. This cannot be undone.`}
        confirmLabel="Delete"
        onCancel={() => setDeleting(null)}
        onConfirm={() => {
          const target = deleting;
          setDeleting(null);
          if (target !== null) {
            void withBusy(async () => {
              await endpoints.deleteClip(target.id);
              toast.success("Clip deleted");
            });
          }
        }}
      />

      {renaming !== null && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          role="dialog"
          aria-modal="true"
        >
          <div className="w-full max-w-sm rounded-xl border border-surface-border bg-surface-raised p-5">
            <h2 className="text-base font-semibold text-slate-100">Rename clip</h2>
            <input
              autoFocus
              value={renameDraft}
              onChange={(event) => setRenameDraft(event.target.value)}
              maxLength={300}
              className="mt-3 w-full rounded-lg border border-surface-border bg-surface px-3 py-2 text-sm focus:border-accent focus:outline-none"
            />
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setRenaming(null)}
                className="rounded-lg px-3 py-2 text-sm text-slate-300 hover:bg-slate-700/50"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={busy}
                onClick={() => {
                  const target = renaming;
                  setRenaming(null);
                  void withBusy(async () => {
                    await endpoints.renameClip(target.id, renameDraft);
                    toast.success("Renamed");
                  });
                }}
                className="rounded-lg bg-accent px-3 py-2 text-sm font-medium text-white hover:bg-accent-soft disabled:opacity-50"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
