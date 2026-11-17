/** Library data loading with filters/sort/pagination. */

import { useCallback, useEffect, useRef, useState } from "react";
import { endpoints, type LibraryQuery } from "@/services/endpoints";
import type { Clip } from "@/types/api";
import { errorMessage } from "@/utils/errors";

export interface LibraryState {
  clips: Clip[];
  total: number;
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

export function useLibrary(
  query: Omit<LibraryQuery, "offset" | "limit"> & { page: number; pageSize: number },
  reloadToken: number,
): LibraryState {
  const [clips, setClips] = useState<Clip[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const requestSeq = useRef(0);

  const { videoId, status, kind, q, sort, page, pageSize } = query;

  const refresh = useCallback(() => {
    setError(null);
    const id = ++requestSeq.current;
    setLoading(true);
    endpoints
      .listLibrary({ videoId, status, kind, q, sort, limit: pageSize, offset: page * pageSize })
      .then((result) => {
        if (requestSeq.current !== id) return;
        setClips(result.items);
        setTotal(result.total);
      })
      .catch((cause: unknown) => {
        if (requestSeq.current !== id) return;
        setError(errorMessage(cause));
      })
      .finally(() => {
        if (requestSeq.current === id) setLoading(false);
      });
  }, [videoId, status, kind, q, sort, page, pageSize]);

  useEffect(() => {
    refresh();
  }, [refresh, reloadToken]);

  return { clips, total, loading, error, refresh };
}
