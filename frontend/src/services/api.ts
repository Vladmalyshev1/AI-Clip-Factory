/** Single API client: typed fetch, unified error envelope, no hidden state. */

import type { ApiErrorBody } from "@/types/api";

export const API_BASE: string =
  import.meta.env.VITE_API_BASE_URL !== undefined &&
  String(import.meta.env.VITE_API_BASE_URL).length > 0
    ? String(import.meta.env.VITE_API_BASE_URL)
    : "";

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly details: Record<string, unknown> | undefined;

  constructor(status: number, code: string, message: string, details?: Record<string, unknown>) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

export class NetworkError extends Error {
  constructor() {
    super("Cannot reach the backend. Is it running?");
    this.name = "NetworkError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/v1${path}`, {
      headers: { "Content-Type": "application/json", ...init?.headers },
      ...init,
    });
  } catch {
    throw new NetworkError();
  }
  if (!response.ok) {
    let code = "UNKNOWN";
    let message = `Request failed (${response.status})`;
    let details: Record<string, unknown> | undefined;
    try {
      const body = (await response.json()) as ApiErrorBody;
      if (typeof body?.error?.code === "string") code = body.error.code;
      if (typeof body?.error?.message === "string") message = body.error.message;
      details = body?.error?.details;
    } catch {
      // Non-JSON error body: keep the defaults.
    }
    throw new ApiError(response.status, code, message, details);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) }),
  put: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "PUT", body: body === undefined ? undefined : JSON.stringify(body) }),
  patch: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "PATCH", body: body === undefined ? undefined : JSON.stringify(body) }),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};

/** Download URL of a rendered clip or export (streams the file). */
export function fileUrl(path: string): string {
  return `${API_BASE}/api/v1${path}`;
}
