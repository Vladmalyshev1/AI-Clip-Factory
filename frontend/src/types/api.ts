/** Types mirroring the backend API schemas (app/schemas/*). */

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
}

export interface Video {
  id: string;
  title: string;
  original_filename: string;
  source_kind: string;
  source_path: string;
  size_bytes: number;
  duration_ms: number | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  has_audio: boolean | null;
  status: string;
  created_at_ms: number;
  updated_at_ms: number;
  thumbnail_url: string | null;
}

export interface ClipError {
  code: string;
  message: string;
}

export interface Clip {
  id: string;
  video_id: string;
  kind: "auto" | "export";
  parent_clip_id: string | null;
  status: "pending" | "processing" | "completed" | "failed" | "canceled";
  title: string | null;
  start_ms: number;
  end_ms: number;
  duration_ms: number;
  score: number | null;
  file_size: number | null;
  width: number | null;
  height: number | null;
  fps: number | null;
  aspect_ratio: string | null;
  output_format: string | null;
  crop_mode: string | null;
  encoding_mode: string | null;
  error: ClipError | null;
  job_id: string | null;
  created_at_ms: number;
  updated_at_ms: number;
}

export interface SubtitleCue {
  start_ms: number;
  end_ms: number;
  text: string;
}

export interface SubtitleSettings {
  enabled: boolean;
  style: "classic" | "bold" | "gaming" | "minimal";
  font_size: number;
  font_name: string;
  primary_colour: string;
  highlight_colour: string;
  position: "bottom" | "center" | "top";
  cues: SubtitleCue[];
}

export interface EditorDraft {
  start_ms: number;
  end_ms: number;
  output_format: "original" | "vertical";
  crop_mode: "crop" | "pad";
  encoding_mode: "accurate" | "fast";
  subtitles: SubtitleSettings;
}

export interface EditorEnvelope {
  draft: EditorDraft;
  transcript_available: boolean;
  suggested_cues: SubtitleCue[];
}

export interface TranscriptResponse {
  transcript_available: boolean;
  cues: SubtitleCue[];
}

export interface ExportList {
  items: Clip[];
  total: number;
}
