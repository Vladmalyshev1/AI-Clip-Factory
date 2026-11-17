/** Library query string builder shared by the page and its tests. */

export interface LibraryFilters {
  videoId?: string;
  status?: string;
  kind?: "auto" | "export";
  q?: string;
  sort?: "date" | "score";
  page?: number;
  pageSize?: number;
}

export function buildLibraryQuery(filters: LibraryFilters): string {
  const params = new URLSearchParams();
  if (filters.videoId) params.set("video_id", filters.videoId);
  if (filters.status) params.set("status", filters.status);
  if (filters.kind) params.set("kind", filters.kind);
  if (filters.q) params.set("q", filters.q);
  params.set("sort", filters.sort ?? "date");
  const pageSize = filters.pageSize ?? 24;
  const page = filters.page ?? 0;
  params.set("limit", String(pageSize));
  params.set("offset", String(page * pageSize));
  return params.toString();
}
