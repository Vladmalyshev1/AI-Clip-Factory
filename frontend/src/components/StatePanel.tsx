/** Centered empty/error/loading state used by all pages. */

interface StatePanelProps {
  title: string;
  hint?: string;
  action?: React.ReactNode;
}

export function StatePanel({ title, hint, action }: StatePanelProps) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-surface-border bg-surface-raised px-6 py-16 text-center">
      <p className="text-sm font-medium text-slate-300">{title}</p>
      {hint !== undefined && hint.length > 0 && (
        <p className="mt-1 max-w-sm text-xs text-slate-500">{hint}</p>
      )}
      {action !== undefined && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function Spinner({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-3 py-16 text-sm text-slate-400">
      <span
        className="h-5 w-5 animate-spin rounded-full border-2 border-slate-600 border-t-accent"
        aria-hidden
      />
      {label}
    </div>
  );
}
