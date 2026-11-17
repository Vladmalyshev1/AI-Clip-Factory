import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Spinner, StatePanel } from "@/components/StatePanel";
import { endpoints } from "@/services/endpoints";
import { fileUrl } from "@/services/api";
import { errorMessage } from "@/utils/errors";
import { formatDuration, formatDate, formatBytes } from "@/utils/format";
import type { Video } from "@/types/api";

export function VideosPage() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [clipCounts, setClipCounts] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void (async () => {
      try {
        const page = await endpoints.listVideos(50);
        setVideos(page.items);
        const counts: Record<string, number> = {};
        await Promise.all(
          page.items.map(async (video) => {
            try {
              const clips = await endpoints.listLibrary({ videoId: video.id, limit: 1 });
              counts[video.id] = clips.total;
            } catch {
              counts[video.id] = 0;
            }
          }),
        );
        setClipCounts(counts);
      } catch (cause) {
        setError(errorMessage(cause));
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) return <Spinner />;
  if (error !== null) {
    return (
      <StatePanel
        title="Could not load videos"
        hint={error}
        action={
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="rounded-lg bg-accent px-3 py-1.5 text-sm text-white"
          >
            Reload
          </button>
        }
      />
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <h1 className="text-lg font-semibold text-slate-100">Source Videos</h1>
      {videos.length === 0 ? (
        <StatePanel
          title="No videos yet"
          hint="Upload or import a recording first — clips are generated from its highlights."
        />
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {videos.map((video) => (
            <div
              key={video.id}
              className="overflow-hidden rounded-xl border border-surface-border bg-surface-raised transition hover:border-accent/40"
            >
              {video.thumbnail_url !== null ? (
                <img
                  src={fileUrl(video.thumbnail_url.replace("/api/v1", ""))}
                  alt=""
                  className="aspect-video w-full object-cover"
                  loading="lazy"
                />
              ) : (
                <div className="flex aspect-video items-center justify-center bg-black/40 text-xs text-slate-600">
                  no preview
                </div>
              )}
              <div className="flex flex-col gap-2 p-3">
                <div className="flex items-start justify-between gap-2">
                  <span className="line-clamp-1 text-sm font-medium text-slate-200">
                    {video.title}
                  </span>
                  <span className="rounded-full bg-slate-700/60 px-2 py-0.5 text-[11px] text-slate-300">
                    {video.status}
                  </span>
                </div>
                <dl className="grid grid-cols-2 gap-x-2 text-xs text-slate-400">
                  <dt>Duration</dt>
                  <dd className="text-right text-slate-300">{formatDuration(video.duration_ms)}</dd>
                  <dt>Size</dt>
                  <dd className="text-right text-slate-300">{formatBytes(video.size_bytes)}</dd>
                  <dt>Clips</dt>
                  <dd className="text-right text-slate-300">{clipCounts[video.id] ?? "…"}</dd>
                  <dt>Imported</dt>
                  <dd className="text-right text-slate-300">{formatDate(video.created_at_ms)}</dd>
                </dl>
                <Link
                  to={`/clips?video=${video.id}`}
                  className="mt-1 rounded-lg bg-accent/20 px-3 py-1.5 text-center text-xs text-accent-soft hover:bg-accent/30"
                >
                  View clips
                </Link>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
