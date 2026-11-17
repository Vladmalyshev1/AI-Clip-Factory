import { useEffect, useState } from "react";

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  body: string;
  confirmLabel?: string;
  destructive?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

/** Modal confirmation for destructive actions (delete, cancel). */
export function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel = "Confirm",
  destructive = true,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) setBusy(false);
  }, [open]);

  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      role="dialog"
      aria-modal="true"
      aria-label={title}
    >
      <div className="w-full max-w-sm rounded-xl border border-surface-border bg-surface-raised p-5 shadow-2xl">
        <h2 className="text-base font-semibold text-slate-100">{title}</h2>
        <p className="mt-2 text-sm text-slate-400">{body}</p>
        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-lg px-3 py-2 text-sm text-slate-300 hover:bg-slate-700/50"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => {
              setBusy(true);
              onConfirm();
            }}
            className={`rounded-lg px-3 py-2 text-sm font-medium text-white disabled:opacity-50 ${
              destructive ? "bg-rose-600 hover:bg-rose-500" : "bg-accent hover:bg-accent-soft"
            }`}
          >
            {busy ? "Working…" : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
