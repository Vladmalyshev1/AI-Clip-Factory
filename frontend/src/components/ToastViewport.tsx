import type { Toast } from "@/hooks/useToasts";
import { toastStore, useToasts } from "@/hooks/useToasts";

const KIND_STYLES: Record<Toast["kind"], string> = {
  success: "border-emerald-500/40 bg-emerald-500/10 text-emerald-200",
  error: "border-rose-500/40 bg-rose-500/10 text-rose-200",
  info: "border-sky-500/40 bg-sky-500/10 text-sky-200",
};

export function ToastViewport() {
  const toasts = useToasts();
  if (toasts.length === 0) return null;
  return (
    <div
      className="fixed bottom-4 right-4 z-50 flex w-80 flex-col gap-2"
      role="status"
      aria-live="polite"
    >
      {toasts.map((toast) => (
        <button
          key={toast.id}
          type="button"
          onClick={() => toastStore.dismiss(toast.id)}
          className={`rounded-lg border px-4 py-3 text-left text-sm shadow-lg transition ${KIND_STYLES[toast.kind]}`}
        >
          {toast.message}
        </button>
      ))}
    </div>
  );
}
