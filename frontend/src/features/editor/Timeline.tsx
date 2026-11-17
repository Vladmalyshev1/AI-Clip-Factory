/** Timeline with draggable start/end handles over the clip window. */

import { useCallback, useEffect, useRef } from "react";

interface TimelineProps {
  durationMs: number;
  startMs: number;
  endMs: number;
  minDurationMs: number;
  maxDurationMs: number;
  positionMs: number;
  onChange: (startMs: number, endMs: number) => void;
  onSeek: (positionMs: number) => void;
}

const HANDLE_WIDTH = 14;

export function Timeline({
  durationMs,
  startMs,
  endMs,
  minDurationMs,
  maxDurationMs,
  positionMs,
  onChange,
  onSeek,
}: TimelineProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const dragging = useRef<"start" | "end" | "seek" | null>(null);

  const toMs = useCallback(
    (clientX: number): number => {
      const track = trackRef.current;
      if (track === null || durationMs <= 0) return 0;
      const rect = track.getBoundingClientRect();
      const fraction = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      return Math.round(fraction * durationMs);
    },
    [durationMs],
  );

  const applyDrag = useCallback(
    (kind: "start" | "end" | "seek", clientX: number) => {
      const ms = toMs(clientX);
      if (kind === "seek") {
        onSeek(Math.min(ms, endMs));
        return;
      }
      if (kind === "start") {
        const next = Math.min(Math.max(0, ms), endMs - minDurationMs);
        onChange(next, endMs);
      } else {
        const next = Math.min(
          Math.max(ms, startMs + minDurationMs),
          Math.min(durationMs, startMs + maxDurationMs),
        );
        onChange(startMs, next);
      }
    },
    [toMs, onSeek, onChange, endMs, startMs, minDurationMs, maxDurationMs, durationMs],
  );

  const onMouseMove = useCallback(
    (event: PointerEvent) => {
      if (dragging.current !== null) applyDrag(dragging.current, event.clientX);
    },
    [applyDrag],
  );

  const stopDrag = useCallback(() => {
    dragging.current = null;
    window.removeEventListener("pointermove", onMouseMove);
    window.removeEventListener("pointerup", stopDrag);
    window.removeEventListener("pointercancel", stopDrag);
  }, [onMouseMove]);

  // Unmounting mid-drag must not leak the window listeners.
  useEffect(() => stopDrag, [stopDrag]);

  const beginDrag = (kind: "start" | "end" | "seek") => (event: React.PointerEvent) => {
    event.preventDefault();
    dragging.current = kind;
    if (kind === "seek") applyDrag("seek", event.clientX);
    window.addEventListener("pointermove", onMouseMove);
    window.addEventListener("pointerup", stopDrag);
    window.addEventListener("pointercancel", stopDrag);
  };

  const pct = (ms: number) => (durationMs > 0 ? (ms / durationMs) * 100 : 0);
  const selectionPct = Math.max(pct(endMs) - pct(startMs), 0.5);

  return (
    <div className="select-none">
      <div
        ref={trackRef}
        onPointerDown={beginDrag("seek")}
        className="relative h-12 cursor-pointer rounded-lg bg-slate-800"
        role="slider"
        aria-label="Timeline"
        aria-valuemin={0}
        aria-valuemax={durationMs}
        aria-valuenow={positionMs}
      >
        {/* waveform-ish decoration */}
        <div className="absolute inset-0 flex items-end overflow-hidden rounded-lg px-1 opacity-40">
          {Array.from({ length: 60 }, (_, index) => (
            <div
              key={index}
              className="mx-px flex-1 rounded-t bg-slate-600"
              style={{ height: `${18 + ((index * 37) % 70)}%` }}
            />
          ))}
        </div>
        {/* selection */}
        <div
          className="absolute inset-y-0 bg-accent/25 ring-1 ring-inset ring-accent/60"
          style={{ left: `${pct(startMs)}%`, width: `${selectionPct}%` }}
        />
        {/* playhead */}
        <div
          className="pointer-events-none absolute inset-y-0 w-0.5 bg-white/90"
          style={{ left: `${pct(positionMs)}%` }}
        />
        {/* handles */}
        <button
          type="button"
          aria-label="Clip start handle"
          onPointerDown={beginDrag("start")}
          className="absolute inset-y-0 z-10 flex cursor-ew-resize items-center justify-center"
          style={{
            left: `calc(${pct(startMs)}% - ${HANDLE_WIDTH / 2}px)`,
            width: HANDLE_WIDTH,
          }}
        >
          <span className="h-8 w-1.5 rounded bg-accent shadow" />
        </button>
        <button
          type="button"
          aria-label="Clip end handle"
          onPointerDown={beginDrag("end")}
          className="absolute inset-y-0 z-10 flex cursor-ew-resize items-center justify-center"
          style={{
            left: `calc(${pct(endMs)}% - ${HANDLE_WIDTH / 2}px)`,
            width: HANDLE_WIDTH,
          }}
        >
          <span className="h-8 w-1.5 rounded bg-accent shadow" />
        </button>
      </div>
      <div className="mt-1 flex justify-between text-[11px] text-slate-500">
        <span>0:00</span>
        <span>{formatPosition(durationMs)}</span>
      </div>
    </div>
  );
}

function formatPosition(ms: number): string {
  const seconds = Math.round(ms / 1000);
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}
