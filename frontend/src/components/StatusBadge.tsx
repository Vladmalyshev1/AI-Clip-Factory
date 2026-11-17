import { STATUS_STYLES } from "@/utils/format";

const LABELS: Record<string, string> = {
  pending: "Pending",
  processing: "Processing",
  completed: "Ready",
  failed: "Failed",
  canceled: "Canceled",
};

export function StatusBadge({ status }: { status: string }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${
        STATUS_STYLES[status] ?? "bg-slate-700/60 text-slate-300"
      }`}
    >
      {LABELS[status] ?? status}
    </span>
  );
}
