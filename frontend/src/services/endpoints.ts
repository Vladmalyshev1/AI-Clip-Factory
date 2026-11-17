/** Typed endpoint helpers — one place that knows every backend URL. */

import { api } from "@/services/api";
import { buildLibraryQuery } from "@/features/library/query";
import type {
  Clip,
  EditorDraft,
  EditorEnvelope,
  ExportList,
  Page,
  TranscriptResponse,
  Video,
} from "@/types/api";

export interface LibraryQuery {
  videoId?: string;
  status?: string;
  kind?: "auto" | "export";
  q?: string;
  sort?: "date" | "score";
  limit?: number;
  offset?: number;
}

function libraryQueryString(query: LibraryQuery): string {
  return buildLibraryQuery(query);
}

export const endpoints = {
  listVideos: (limit = 50, offset = 0) =>
    api.get<Page<Video>>(`/videos?limit=${limit}&offset=${offset}`),
  getVideo: (videoId: string) => api.get<Video>(`/videos/${videoId}`),
  listLibrary: (query: LibraryQuery) => api.get<Page<Clip>>(`/clips?${libraryQueryString(query)}`),
  getClip: (clipId: string) => api.get<Clip>(`/clips/${clipId}`),
  renameClip: (clipId: string, title: string) =>
    api.patch<Clip>(`/clips/${clipId}`, { title }),
  deleteClip: (clipId: string) => api.delete<void>(`/clips/${clipId}`),
  retryClip: (clipId: string) => api.post<Clip>(`/clips/${clipId}/retry`),
  cancelClip: (clipId: string) => api.post<Clip>(`/clips/${clipId}/cancel`),
  getEditor: (clipId: string) => api.get<EditorEnvelope>(`/clips/${clipId}/editor`),
  putEditor: (clipId: string, draft: Partial<EditorDraft>) =>
    api.put<EditorDraft>(`/clips/${clipId}/editor`, draft),
  getTranscript: (clipId: string) =>
    api.get<TranscriptResponse>(`/clips/${clipId}/transcript`),
  createExport: (clipId: string, body?: Partial<EditorDraft>) =>
    api.post<ExportList>(`/clips/${clipId}/exports`, body ?? {}),
  listExports: (clipId: string) => api.get<ExportList>(`/clips/${clipId}/exports`),
  getExport: (exportId: string) => api.get<Clip>(`/exports/${exportId}`),
  cancelExport: (exportId: string) => api.post<Clip>(`/exports/${exportId}/cancel`),
  retryExport: (exportId: string) => api.post<Clip>(`/exports/${exportId}/retry`),
  deleteExport: (exportId: string) => api.delete<void>(`/exports/${exportId}`),
};
