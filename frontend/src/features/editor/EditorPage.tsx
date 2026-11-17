/** Clip editor page: player, timeline, draft editing, export. */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Spinner, StatePanel } from "@/components/StatePanel";
import { Player } from "./Player";
import { Timeline } from "./Timeline";
import { SubtitlesPanel } from "./SubtitlesPanel";
import { ExportPanel } from "./ExportPanel";
import { endpoints } from "@/services/endpoints";
import { toast } from "@/hooks/useToasts";
import { errorMessage } from "@/utils/errors";
import { formatTimestamp } from "@/utils/format";
import type { EditorDraft, SubtitleCue } from "@/types/api";

const MIN_DURATION_MS = 1_000;
const MAX_DURATION_MS = 600_000;

export function EditorPage() {
  const { clipId = "" } = useParams();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [draft, setDraft] = useState<EditorDraft | null>(null);
  const [savedDraft, setSavedDraft] = useState<EditorDraft | null>(null);
  const [videoTitle, setVideoTitle] = useState("");
  const [videoId, setVideoId] = useState("");
  const [videoDurationMs, setVideoDurationMs] = useState<number | null>(null);
  const [clipTitle, setClipTitle] = useState("");
  const [clipKind, setClipKind] = useState<"auto" | "export">("auto");
  const [score, setScore] = useState<number | null>(null);
  const [suggestedCues, setSuggestedCues] = useState<SubtitleCue[]>([]);
  const [transcriptAvailable, setTranscriptAvailable] = useState(false);
  const [positionMs, setPositionMs] = useState(0);
  const [seekRequest, setSeekRequest] = useState<number | null>(null);
  const [saving, setSaving] = useState(false);
  const dirtyRef = useRef(false);

  const dirty = useMemo(
    () => draft !== null && savedDraft !== null && JSON.stringify(draft) !== JSON.stringify(savedDraft),
    [draft, savedDraft],
  );

  useEffect(() => {
    setLoading(true);
    setError(null);
    void (async () => {
      try {
        const envelope = await endpoints.getEditor(clipId);
        setDraft(envelope.draft);
        setSavedDraft(envelope.draft);
        setSuggestedCues(envelope.suggested_cues);
        setTranscriptAvailable(envelope.transcript_available);
        const clip = await endpoints.getClip(clipId);
        setClipTitle(clip.title ?? `Clip ${clip.id.slice(0, 8)}`);
        setClipKind(clip.kind);
        setScore(clip.score);
        const video = await endpoints.getVideo(clip.video_id);
        setVideoId(video.id);
        setVideoTitle(video.title);
        setVideoDurationMs(video.duration_ms);
      } catch (cause) {
        setError(errorMessage(cause));
      } finally {
        setLoading(false);
      }
    })();
  }, [clipId]);

  const saveDraft = useCallback(
    async (next: EditorDraft): Promise<void> => {
      setSaving(true);
      try {
        const saved = await endpoints.putEditor(clipId, next);
        setSavedDraft(saved);
        setDraft(saved);
        dirtyRef.current = false;
        toast.success("Draft saved");
      } catch (cause) {
        toast.error(errorMessage(cause));
        throw cause;
      } finally {
        setSaving(false);
      }
    },
    [clipId],
  );

  if (loading) return <Spinner label="Loading editor…" />;
  if (error !== null || draft === null) {
    return (
      <StatePanel
        title="Could not open the editor"
        hint={error ?? "Unknown error"}
        action={
          <Link to="/clips" className="rounded-lg bg-accent px-3 py-1.5 text-sm text-white">
            Back to library
          </Link>
        }
      />
    );
  }

  const durationMs = videoDurationMs ?? draft.end_ms;
  const windowMs = draft.end_ms - draft.start_ms;

  const changeBounds = (startMs: number, endMs: number) => {
    setDraft({ ...draft, start_ms: startMs, end_ms: endMs });
    dirtyRef.current = true;
  };

  const nudge = (which: "start" | "end", deltaMs: number) => {
    if (which === "start") {
      const next = Math.min(Math.max(0, draft.start_ms + deltaMs), draft.end_ms - MIN_DURATION_MS);
      changeBounds(next, draft.end_ms);
    } else {
      const limit = Math.min(durationMs, draft.start_ms + MAX_DURATION_MS);
      const next = Math.min(Math.max(draft.end_ms + deltaMs, draft.start_ms + MIN_DURATION_MS), limit);
      changeBounds(draft.start_ms, next);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-lg font-semibold text-slate-100">{clipTitle}</h1>
          <p className="text-xs text-slate-500">
            from "{videoTitle}" · score {score === null ? "—" : score.toFixed(2)}
          </p>
        </div>
        <Link
          to="/clips"
          className="rounded-lg px-3 py-1.5 text-sm text-slate-400 hover:bg-slate-700/40 hover:text-slate-200"
        >
          ← Library
        </Link>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="flex flex-col gap-4">
          <Player
            videoId={videoId}
            startMs={draft.start_ms}
            endMs={draft.end_ms}
            subtitles={draft.subtitles}
            onPositionChange={setPositionMs}
            seekRequestMs={seekRequest}
            onSeekHandled={() => setSeekRequest(null)}
          />
          <Timeline
            durationMs={durationMs}
            startMs={draft.start_ms}
            endMs={draft.end_ms}
            minDurationMs={MIN_DURATION_MS}
            maxDurationMs={MAX_DURATION_MS}
            positionMs={positionMs}
            onChange={changeBounds}
            onSeek={(ms) => setSeekRequest(ms)}
          />
          <div className="flex flex-wrap items-center gap-3 rounded-xl border border-surface-border bg-surface-raised p-3 text-xs">
            <span className="font-mono text-slate-300">
              {formatTimestamp(draft.start_ms)} → {formatTimestamp(draft.end_ms)}
            </span>
            <span className="rounded-full bg-accent/20 px-2 py-0.5 text-accent-soft">
              {formatTimestamp(windowMs)} selected
            </span>
            <div className="flex items-center gap-1">
              <button type="button" onClick={() => nudge("start", -250)} className="rounded bg-slate-700/50 px-2 py-1 hover:bg-slate-600/50" title="Start −250 ms">⟨−</button>
              <button type="button" onClick={() => nudge("start", 250)} className="rounded bg-slate-700/50 px-2 py-1 hover:bg-slate-600/50" title="Start +250 ms">+⟩</button>
              <button type="button" onClick={() => nudge("end", -250)} className="rounded bg-slate-700/50 px-2 py-1 hover:bg-slate-600/50" title="End −250 ms">−⟨</button>
              <button type="button" onClick={() => nudge("end", 250)} className="rounded bg-slate-700/50 px-2 py-1 hover:bg-slate-600/50" title="End +250 ms">+⟩</button>
            </div>
            <button
              type="button"
              onClick={() => setSeekRequest(draft.start_ms)}
              className="rounded bg-slate-700/50 px-2 py-1 text-slate-300 hover:bg-slate-600/50"
            >
              Preview clip
            </button>
            <span className="ml-auto">
              {dirty && (
                <button
                  type="button"
                  disabled={saving}
                  onClick={() => void saveDraft(draft)}
                  className="rounded-lg bg-accent px-3 py-1.5 font-medium text-white hover:bg-accent-soft disabled:opacity-50"
                >
                  {saving ? "Saving…" : "Save draft"}
                </button>
              )}
            </span>
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <section className="flex flex-col gap-3 rounded-xl border border-surface-border bg-surface-raised p-4">
            <h2 className="text-sm font-semibold text-slate-200">Format</h2>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <label className="flex flex-col gap-1">
                <span className="text-slate-400">Output</span>
                <select
                  value={draft.output_format}
                  onChange={(event) =>
                    setDraft({
                      ...draft,
                      output_format: event.target.value as EditorDraft["output_format"],
                    })
                  }
                  className="rounded-lg border border-surface-border bg-surface px-2 py-1.5 focus:border-accent focus:outline-none"
                >
                  <option value="vertical">Vertical 9:16</option>
                  <option value="original">Original</option>
                </select>
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-slate-400">Crop / pad</span>
                <select
                  value={draft.crop_mode}
                  disabled={draft.output_format === "original"}
                  onChange={(event) =>
                    setDraft({ ...draft, crop_mode: event.target.value as EditorDraft["crop_mode"] })
                  }
                  className="rounded-lg border border-surface-border bg-surface px-2 py-1.5 focus:border-accent focus:outline-none disabled:opacity-40"
                >
                  <option value="pad">Pad (blur bg)</option>
                  <option value="crop">Crop fill</option>
                </select>
              </label>
              <label className="flex flex-col gap-1">
                <span className="text-slate-400">Encoding</span>
                <select
                  value={draft.encoding_mode}
                  onChange={(event) =>
                    setDraft({
                      ...draft,
                      encoding_mode: event.target.value as EditorDraft["encoding_mode"],
                    })
                  }
                  className="rounded-lg border border-surface-border bg-surface px-2 py-1.5 focus:border-accent focus:outline-none"
                >
                  <option value="accurate">Accurate</option>
                  <option value="fast">Fast (original only)</option>
                </select>
              </label>
            </div>
            {dirty && (
              <button
                type="button"
                disabled={saving}
                onClick={() => void saveDraft(draft)}
                className="rounded-lg bg-slate-700/50 px-3 py-1.5 text-xs text-slate-200 hover:bg-slate-600/50 disabled:opacity-50"
              >
                {saving ? "Saving…" : "Save format changes"}
              </button>
            )}
          </section>

          <SubtitlesPanel
            draft={draft}
            onChange={(next) => setDraft(next)}
            suggestedCues={suggestedCues}
            transcriptAvailable={transcriptAvailable}
          />

          <ExportPanel
            clipId={clipId}
            draft={draft}
            onSaveAndExport={() => saveDraft(draft)}
          />
          {clipKind === "export" && (
            <p className="rounded-lg bg-slate-700/40 px-3 py-2 text-xs text-slate-400">
              This clip is an export of another clip. Exports are cut from generated
              clips only — open the source clip to export it again.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
