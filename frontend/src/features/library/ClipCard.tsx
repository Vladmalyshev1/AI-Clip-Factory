import { Link } from "react-router-dom";
import { StatusBadge } from "@/components/StatusBadge";
import type { Clip } from "@/types/api";
import { fileUrl } from "@/services/api";
import { isDownloadable, formatDuration, formatScore, formatDate } from "@/utils/format";
interface ClipCardProps {
  clip: Clip;
  onRename: (clip: Clip) => void;
  onDelete: (clip: Clip) => void;
  onRetry: (clip: Clip) => void;
  busy: boolean;
}

const KIND_LABEL: Record<Clip["kind"], string> = {
  auto: "Auto",
  export: "Export",
};

export function ClipCard({ clip, onRename, onDelete, onRetry, busy }: ClipCardProps) {
  const downloadable = isDownloadable(clip.status);
  const frame =
    clip.width !== null && clip.height !== null ? `${clip.width}×${clip.height}` : null;

  return (
    <div className="group flex flex-col overflow-hidden rounded-xl border border-surface-border bg-surface-raised transition hover:border-accent/40">
      <Link
        to={`/clips/${clip.id}`}
        className="relative flex aspect-video items-center justify-center bg-black/40"
      >
        {downloadable ? (
          <span className="text-xs text-slate-500 group-hover:text-accent-soft">
            ▶ open editor
          </span>
        ) : (
          <span className="text-xs text-slate-600">{clip.status}…</span>
        )}
        <span className="absolute right-2 top-2 rounded-full bg-black/60 px-2 py-0.5 text-[11px] text-slate-300">
          {KIND_LABEL[clip.kind]}
        </span>
      </Link>
      <div className="flex flex-1 flex-col gap-2 p-3">
        <div className="flex items-start justify-between gap-2">
          <Link
            to={`/clips/${clip.id}`}
            className="line-clamp-2 text-sm font-medium text-slate-200 hover:text-accent-soft"
          >
            {clip.title ?? `Clip ${clip.id.slice(0, 8)}`}
          </Link>
          <StatusBadge status={clip.status} />
        </div>
        <dl className="grid grid-cols-2 gap-x-2 gap-y-1 text-xs text-slate-400">
          <dt>Score</dt>
          <dd className="text-right text-slate-300">{formatScore(clip.score)}</dd>
          <dt>Duration</dt>
          <dd className="text-right text-slate-300">{formatDuration(clip.duration_ms)}</dd>
          <dt>Range</dt>
          <dd className="text-right text-slate-300">
            {formatDuration(clip.start_ms)} – {formatDuration(clip.end_ms)}
          </dd>
          <dt>Frame</dt>
          <dd className="text-right text-slate-300">{frame ?? "—"}</dd>
          <dt>Created</dt>
          <dd className="text-right text-slate-300">{formatDate(clip.created_at_ms)}</dd>
        </dl>
        {clip.error !== null && (
          <p className="rounded-lg bg-rose-500/10 px-2 py-1 text-xs text-rose-300">
            {clip.error.code}: {clip.error.message}
          </p>
        )}
        <div className="mt-auto flex flex-wrap items-center gap-1.5 pt-1">
          {downloadable && (
            <a
              href={fileUrl(`/clips/${clip.id}/file`)}
              download
              className="rounded-lg bg-slate-700/50 px-2.5 py-1.5 text-xs text-slate-200 hover:bg-slate-600/50"
            >
              Download
            </a>
          )}
          <Link
            to={`/clips/${clip.id}`}
            className="rounded-lg bg-accent/20 px-2.5 py-1.5 text-xs text-accent-soft hover:bg-accent/30"
          >
            Edit
          </Link>
          <button
            type="button"
            disabled={busy}
            onClick={() => onRename(clip)}
            className="rounded-lg px-2.5 py-1.5 text-xs text-slate-400 hover:bg-slate-700/40 hover:text-slate-200 disabled:opacity-40"
          >
            Rename
          </button>
          {(clip.status === "failed" || clip.status === "canceled") && (
            <button
              type="button"
              disabled={busy}
              onClick={() => onRetry(clip)}
              className="rounded-lg px-2.5 py-1.5 text-xs text-amber-300 hover:bg-amber-500/10 disabled:opacity-40"
            >
              Retry
            </button>
          )}
          {(clip.status === "pending" || clip.status === "processing") && (
            <button
              type="button"
              disabled={busy}
              onClick={() => onRetry(clip)}
              title="Cancel render"
              className="rounded-lg px-2.5 py-1.5 text-xs text-sky-300 hover:bg-sky-500/10 disabled:opacity-40"
            >
              Cancel
            </button>
          )}
          <button
            type="button"
            disabled={busy}
            onClick={() => onDelete(clip)}
            className="ml-auto rounded-lg px-2.5 py-1.5 text-xs text-rose-300 hover:bg-rose-500/10 disabled:opacity-40"
          >
            Delete
          </button>
        </div>
      </div>
    </div>
  );
}
