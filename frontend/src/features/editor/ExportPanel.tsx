/** Export panel: format choice, subtitle toggle, live export status, download. */

import { useCallback, useEffect, useState } from "react";
import { StatusBadge } from "@/components/StatusBadge";
import { endpoints } from "@/services/endpoints";
import { fileUrl } from "@/services/api";
import { toast } from "@/hooks/useToasts";
import { errorMessage } from "@/utils/errors";
import { formatBytes, isDownloadable } from "@/utils/format";
import type { Clip, EditorDraft } from "@/types/api";

interface ExportPanelProps {
  clipId: string;
  draft: EditorDraft;
  onSaveAndExport: () => Promise<void>;
}

const POLL_INTERVAL_MS = 1500;

export function ExportPanel({ clipId, draft, onSaveAndExport }: ExportPanelProps) {
  const [exports, setExports] = useState<Clip[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const list = await endpoints.listExports(clipId);
    setExports(list.items);
    setActiveId((current) => {
      if (current !== null && list.items.some((item) => item.id === current)) return current;
      const active = list.items.find(
        (item) => item.status === "pending" || item.status === "processing",
      );
      return active?.id ?? null;
    });
  }, [clipId]);

  useEffect(() => {
    void refresh().catch((cause: unknown) => setError(errorMessage(cause)));
  }, [refresh]);

  // Poll only while an export is pending/processing: real statuses, no fake %.
  useEffect(() => {
    if (activeId === null) return;
    const timer = setInterval(() => {
      void refresh().catch(() => undefined);
    }, POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [activeId, refresh]);

  const active = exports.find((item) => item.id === activeId) ?? null;

  const startExport = () => {
    setBusy(true);
    setError(null);
    void (async () => {
      try {
        await onSaveAndExport();
        const list = await endpoints.createExport(clipId);
        const latest = list.items[0];
        if (latest !== undefined) {
          setActiveId(latest.id);
          if (latest.status === "pending") toast.info("Export queued");
        }
        await refresh();
        toast.success("Export started");
      } catch (cause) {
        const message = errorMessage(cause);
        setError(message);
        toast.error(message);
      } finally {
        setBusy(false);
      }
    })();
  };

  const withBusy = (action: () => Promise<void>, successMessage: string) => {
    setBusy(true);
    void (async () => {
      try {
        await action();
        await refresh();
        toast.success(successMessage);
      } catch (cause) {
        toast.error(errorMessage(cause));
      } finally {
        setBusy(false);
      }
    })();
  };

  return (
    <section className="flex flex-col gap-3 rounded-xl border border-surface-border bg-surface-raised p-4">
      <h2 className="text-sm font-semibold text-slate-200">Export</h2>

      <div className="flex gap-2 rounded-lg bg-surface p-1 text-xs">
        <span
          className={`flex-1 rounded-md px-2 py-1.5 text-center ${
            draft.output_format === "vertical" ? "bg-accent/25 text-accent-soft" : "text-slate-400"
          }`}
        >
          Vertical 9:16 · 1080×1920
        </span>
        <span
          className={`flex-1 rounded-md px-2 py-1.5 text-center ${
            draft.output_format === "original" ? "bg-accent/25 text-accent-soft" : "text-slate-400"
          }`}
        >
          Original geometry
        </span>
      </div>
      <p className="text-xs text-slate-500">
        {draft.subtitles.enabled
          ? `Subtitles: ON (${draft.subtitles.style}, ${draft.subtitles.cues.length} cues)`
          : "Subtitles: off"}{" "}
        · {draft.encoding_mode === "fast" ? "Fast (keyframe-snapped)" : "Accurate re-encode"}
      </p>

      <button
        type="button"
        disabled={busy}
        onClick={startExport}
        className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:bg-accent-soft disabled:opacity-50"
      >
        {busy ? "Working…" : "Save & export MP4"}
      </button>
      {error !== null && <p className="text-xs text-rose-300">{error}</p>}

      {active !== null && (
        <div className="rounded-lg border border-surface-border bg-surface p-3 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-slate-300">Latest export</span>
            <StatusBadge status={active.status} />
          </div>
          <p className="mt-1 text-slate-500">
            {active.status === "pending" && "Waiting in the render queue…"}
            {active.status === "processing" && "FFmpeg is rendering…"}
            {active.status === "completed" && `Done · ${formatBytes(active.file_size)}`}
            {active.status === "failed" && active.error !== null && (
              <span className="text-rose-300">
                {active.error.code}: {active.error.message}
              </span>
            )}
            {active.status === "canceled" && "Cancelled"}
          </p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {isDownloadable(active.status) && (
              <a
                href={fileUrl(`/exports/${active.id}/file`)}
                download
                className="rounded-lg bg-emerald-500/20 px-2.5 py-1.5 text-emerald-300 hover:bg-emerald-500/30"
              >
                Download MP4
              </a>
            )}
            {(active.status === "pending" || active.status === "processing") && (
              <button
                type="button"
                disabled={busy}
                onClick={() =>
                  withBusy(async () => {
                    await endpoints.cancelExport(active.id);
                    setActiveId(null);
                  }, "Export cancelled")
                }
                className="rounded-lg bg-sky-500/20 px-2.5 py-1.5 text-sky-300 hover:bg-sky-500/30 disabled:opacity-40"
              >
                Cancel
              </button>
            )}
            {(active.status === "failed" || active.status === "canceled") && (
              <button
                type="button"
                disabled={busy}
                onClick={() =>
                  withBusy(async () => {
                    await endpoints.retryExport(active.id);
                  }, "Export re-queued")
                }
                className="rounded-lg bg-amber-500/20 px-2.5 py-1.5 text-amber-300 hover:bg-amber-500/30 disabled:opacity-40"
              >
                Retry
              </button>
            )}
          </div>
        </div>
      )}

      {exports.length > 0 && (
        <details className="text-xs text-slate-400">
          <summary className="cursor-pointer select-none">Export history ({exports.length})</summary>
          <ul className="mt-2 flex flex-col gap-1">
            {exports.map((item) => (
              <li key={item.id} className="flex items-center gap-2">
                <StatusBadge status={item.status} />
                <span className="text-slate-500">{formatBytes(item.file_size)}</span>
                {isDownloadable(item.status) && (
                  <a
                    href={fileUrl(`/exports/${item.id}/file`)}
                    download
                    className="text-accent-soft hover:underline"
                  >
                    download
                  </a>
                )}
                <button
                  type="button"
                  disabled={busy}
                  onClick={() =>
                    withBusy(() => endpoints.deleteExport(item.id), "Export deleted")
                  }
                  className="ml-auto text-rose-300 hover:text-rose-200 disabled:opacity-40"
                >
                  delete
                </button>
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}
