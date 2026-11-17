/** Subtitle editing panel: cues, style presets and colours. */

import type { EditorDraft, SubtitleCue } from "@/types/api";
import { formatTimestamp } from "@/utils/format";

const STYLE_PRESETS = ["classic", "bold", "gaming", "minimal"] as const;
const POSITIONS = ["bottom", "center", "top"] as const;

interface SubtitlesPanelProps {
  draft: EditorDraft;
  onChange: (draft: EditorDraft) => void;
  suggestedCues: SubtitleCue[];
  transcriptAvailable: boolean;
}

export function SubtitlesPanel({
  draft,
  onChange,
  suggestedCues,
  transcriptAvailable,
}: SubtitlesPanelProps) {
  const subs = draft.subtitles;
  const windowMs = draft.end_ms - draft.start_ms;

  const patch = (partial: Partial<EditorDraft["subtitles"]>) => {
    onChange({ ...draft, subtitles: { ...subs, ...partial } });
  };

  const updateCue = (index: number, partial: Partial<SubtitleCue>) => {
    const cues = subs.cues.map((cue, position) =>
      position === index ? { ...cue, ...partial } : cue,
    );
    patch({ cues });
  };

  return (
    <section className="flex flex-col gap-3 rounded-xl border border-surface-border bg-surface-raised p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-200">Subtitles</h2>
        <label className="flex items-center gap-2 text-xs text-slate-400">
          <input
            type="checkbox"
            checked={subs.enabled}
            onChange={(event) => patch({ enabled: event.target.checked })}
            className="accent-[var(--color-accent)]"
          />
          Enabled
        </label>
      </div>

      {!subs.enabled && (
        <p className="text-xs text-slate-500">
          Subtitles are burned into the exported MP4 only when enabled here.
        </p>
      )}

      <div className="grid grid-cols-2 gap-2 text-xs">
        <label className="flex flex-col gap-1">
          <span className="text-slate-400">Style</span>
          <select
            value={subs.style}
            onChange={(event) => patch({ style: event.target.value as typeof subs.style })}
            className="rounded-lg border border-surface-border bg-surface px-2 py-1.5 focus:border-accent focus:outline-none"
          >
            {STYLE_PRESETS.map((preset) => (
              <option key={preset} value={preset}>
                {preset[0].toUpperCase() + preset.slice(1)}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-slate-400">Position</span>
          <select
            value={subs.position}
            onChange={(event) => patch({ position: event.target.value as typeof subs.position })}
            className="rounded-lg border border-surface-border bg-surface px-2 py-1.5 focus:border-accent focus:outline-none"
          >
            {POSITIONS.map((position) => (
              <option key={position} value={position}>
                {position[0].toUpperCase() + position.slice(1)}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-slate-400">Font size</span>
          <input
            type="number"
            min={16}
            max={200}
            value={subs.font_size}
            onChange={(event) => patch({ font_size: Number(event.target.value) })}
            className="rounded-lg border border-surface-border bg-surface px-2 py-1.5 focus:border-accent focus:outline-none"
          />
        </label>
        <div className="flex items-end gap-2">
          <label className="flex flex-col gap-1">
            <span className="text-slate-400">Text</span>
            <input
              type="color"
              value={subs.primary_colour}
              onChange={(event) => patch({ primary_colour: event.target.value })}
              className="h-8 w-12 rounded border border-surface-border bg-surface"
            />
          </label>
          <label className="flex flex-col gap-1">
            <span className="text-slate-400">Highlight</span>
            <input
              type="color"
              value={subs.highlight_colour}
              onChange={(event) => patch({ highlight_colour: event.target.value })}
              className="h-8 w-12 rounded border border-surface-border bg-surface"
            />
          </label>
        </div>
      </div>

      <div className="flex items-center justify-between pt-1 text-xs">
        <span className="text-slate-400">
          Cues ({subs.cues.length}) · phrase-level, from the M3 transcript
        </span>
        {transcriptAvailable && subs.cues.length === 0 && suggestedCues.length > 0 && (
          <button
            type="button"
            onClick={() => patch({ cues: suggestedCues, enabled: true })}
            className="rounded-lg bg-accent/20 px-2 py-1 text-accent-soft hover:bg-accent/30"
          >
            Use transcript cues
          </button>
        )}
        {!transcriptAvailable && (
          <span className="text-slate-500">No completed transcript for this video</span>
        )}
      </div>

      <div className="flex max-h-64 flex-col gap-2 overflow-y-auto pr-1">
        {subs.cues.map((cue, index) => (
          <div
            key={`${cue.start_ms}-${index}`}
            className="rounded-lg border border-surface-border bg-surface p-2"
          >
            <div className="flex items-center gap-2 text-[11px] text-slate-500">
              <span className="font-mono">{formatTimestamp(cue.start_ms)}</span>
              <span>→</span>
              <span className="font-mono">{formatTimestamp(cue.end_ms)}</span>
              <button
                type="button"
                onClick={() => patch({ cues: subs.cues.filter((_, i) => i !== index) })}
                className="ml-auto text-rose-300 hover:text-rose-200"
                aria-label={`Delete cue ${index + 1}`}
              >
                ✕
              </button>
            </div>
            <textarea
              value={cue.text}
              rows={2}
              onChange={(event) => updateCue(index, { text: event.target.value })}
              className="mt-1 w-full resize-none rounded-md border border-surface-border bg-surface-raised px-2 py-1 text-sm focus:border-accent focus:outline-none"
            />
            <div className="mt-1 flex gap-2">
              <input
                type="number"
                min={0}
                max={windowMs}
                value={cue.start_ms}
                onChange={(event) => updateCue(index, { start_ms: Number(event.target.value) })}
                className="w-full rounded-md border border-surface-border bg-surface-raised px-2 py-1 font-mono text-[11px] focus:border-accent focus:outline-none"
                aria-label={`Cue ${index + 1} start (ms)`}
              />
              <input
                type="number"
                min={0}
                max={windowMs}
                value={cue.end_ms}
                onChange={(event) => updateCue(index, { end_ms: Number(event.target.value) })}
                className="w-full rounded-md border border-surface-border bg-surface-raised px-2 py-1 font-mono text-[11px] focus:border-accent focus:outline-none"
                aria-label={`Cue ${index + 1} end (ms)`}
              />
            </div>
          </div>
        ))}
      </div>

      <button
        type="button"
        onClick={() =>
          patch({
            cues: [
              ...subs.cues,
              { start_ms: 0, end_ms: Math.min(1500, windowMs), text: "New cue" },
            ],
            enabled: true,
          })
        }
        className="rounded-lg border border-dashed border-surface-border px-2 py-1.5 text-xs text-slate-400 hover:border-accent/50 hover:text-accent-soft"
      >
        + Add cue
      </button>
    </section>
  );
}
