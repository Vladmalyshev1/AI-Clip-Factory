/** Video player over the immutable source with clip-relative overlay preview. */

import { useEffect, useRef, useState } from "react";
import { fileUrl } from "@/services/api";
import type { SubtitleSettings } from "@/types/api";
import { formatTimestamp } from "@/utils/format";

interface PlayerProps {
  videoId: string;
  startMs: number;
  endMs: number;
  subtitles: SubtitleSettings | null;
  onPositionChange: (positionMs: number) => void;
  seekRequestMs: number | null;
  onSeekHandled: () => void;
}

const POSITION_TOP = "top-16";
const POSITION_CENTER = "top-1/2 -translate-y-1/2";
const POSITION_BOTTOM = "bottom-16";

export function Player({
  videoId,
  startMs,
  endMs,
  subtitles,
  onPositionChange,
  seekRequestMs,
  onSeekHandled,
}: PlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [volume, setVolume] = useState(1);
  const [positionMs, setPositionMs] = useState(startMs);
  const [error, setError] = useState<string | null>(null);
  const [loopWithinClip, setLoopWithinClip] = useState(true);

  // External seek requests (timeline clicks, editor buttons).
  useEffect(() => {
    if (seekRequestMs === null) return;
    const video = videoRef.current;
    if (video !== null) {
      video.currentTime = seekRequestMs / 1000;
      setPositionMs(seekRequestMs);
    }
    onSeekHandled();
  }, [seekRequestMs, onSeekHandled]);

  const togglePlay = () => {
    const video = videoRef.current;
    if (video === null) return;
    if (video.paused) {
      void video.play();
    } else {
      video.pause();
    }
  };

  const cueAt = (ms: number): string | null => {
    if (subtitles === null || !subtitles.enabled) return null;
    const relative = ms - startMs;
    const cue = subtitles.cues.find((item) => relative >= item.start_ms && relative < item.end_ms);
    return cue === undefined ? null : cue.text;
  };

  const activeCue = cueAt(positionMs);

  return (
    <div className="flex flex-col gap-3">
      <div className="relative overflow-hidden rounded-xl bg-black">
        <video
          ref={videoRef}
          src={fileUrl(`/videos/${videoId}/source`)}
          preload="metadata"
          className="max-h-[60vh] w-full"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onError={() => setError("The source file is unavailable. Re-import the video.")}
          onTimeUpdate={(event) => {
            const ms = event.currentTarget.currentTime * 1000;
            setPositionMs(ms);
            onPositionChange(ms);
            const video = event.currentTarget;
            if (loopWithinClip && ms >= endMs && !video.paused) {
              video.currentTime = startMs / 1000;
            }
          }}
        />
        {activeCue !== null && subtitles !== null && (
          <div
            className={`pointer-events-none absolute inset-x-0 flex justify-center px-6 ${
              subtitles.position === "top"
                ? POSITION_TOP
                : subtitles.position === "center"
                  ? POSITION_CENTER
                  : POSITION_BOTTOM
            }`}
          >
            <span
              className="rounded-lg bg-black/60 px-3 py-1.5 text-center leading-tight text-white"
              style={{
                fontSize: `${Math.max(12, Math.round(subtitles.font_size / 3))}px`,
                fontWeight: subtitles.style === "minimal" ? 400 : 700,
                textShadow: "0 2px 6px rgba(0,0,0,0.9)",
              }}
            >
              {activeCue}
            </span>
          </div>
        )}
        {error !== null && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/80 px-6 text-center text-sm text-rose-300">
            {error}
          </div>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <button
          type="button"
          onClick={togglePlay}
          className="rounded-lg bg-accent px-4 py-1.5 font-medium text-white hover:bg-accent-soft"
        >
          {playing ? "Pause" : "Play"}
        </button>
        <button
          type="button"
          onClick={() => {
            const video = videoRef.current;
            if (video !== null) video.currentTime = startMs / 1000;
          }}
          className="rounded-lg bg-slate-700/50 px-3 py-1.5 text-slate-200 hover:bg-slate-600/50"
        >
          ⏮ clip start
        </button>
        <label className="flex items-center gap-2 text-xs text-slate-400">
          Volume
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={volume}
            onChange={(event) => {
              const next = Number(event.target.value);
              setVolume(next);
              const video = videoRef.current;
              if (video !== null) video.volume = next;
            }}
            className="w-24 accent-[var(--color-accent)]"
          />
        </label>
        <label className="flex items-center gap-1.5 text-xs text-slate-400">
          <input
            type="checkbox"
            checked={loopWithinClip}
            onChange={(event) => setLoopWithinClip(event.target.checked)}
            className="accent-[var(--color-accent)]"
          />
          Loop inside clip
        </label>
        <span className="ml-auto font-mono text-xs text-slate-400">
          {formatTimestamp(positionMs)} / {formatTimestamp(endMs)}
        </span>
      </div>
    </div>
  );
}
