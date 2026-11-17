/** Error-to-message helper used by every page. */

import { ApiError, NetworkError } from "@/services/api";

export function errorMessage(error: unknown): string {
  if (error instanceof NetworkError) return error.message;
  if (error instanceof ApiError) return `${error.message} (${error.code})`;
  if (error instanceof Error) return error.message;
  return String(error);
}
