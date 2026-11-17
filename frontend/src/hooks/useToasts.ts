/** Tiny toast store: success/error notifications without a dependency. */

import { useSyncExternalStore } from "react";

export interface Toast {
  id: number;
  kind: "success" | "error" | "info";
  message: string;
}

let toasts: Toast[] = [];
let nextId = 1;
const listeners = new Set<() => void>();

function emit(): void {
  listeners.forEach((listener) => listener());
}

function push(kind: Toast["kind"], message: string): void {
  const toast: Toast = { id: nextId++, kind, message };
  toasts = [...toasts, toast];
  emit();
  setTimeout(() => {
    toasts = toasts.filter((item) => item.id !== toast.id);
    emit();
  }, 4000);
}

export const toastStore = {
  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  getSnapshot(): Toast[] {
    return toasts;
  },
  success(message: string): void {
    push("success", message);
  },
  error(message: string): void {
    push("error", message);
  },
  info(message: string): void {
    push("info", message);
  },
  dismiss(id: number): void {
    toasts = toasts.filter((item) => item.id !== id);
    emit();
  },
};

export function useToasts(): Toast[] {
  return useSyncExternalStore(toastStore.subscribe, toastStore.getSnapshot);
}

export const toast = toastStore;
