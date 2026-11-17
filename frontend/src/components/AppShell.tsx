import { Link, NavLink } from "react-router-dom";

const TABS = [
  { to: "/videos", label: "Videos" },
  { to: "/clips", label: "Clip Library" },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-full flex-col">
      <header className="border-b border-surface-border bg-surface-raised/60">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
          <Link to="/videos" className="flex items-center gap-2 font-semibold text-slate-100">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent text-sm">
              ✂
            </span>
            AI Clip Factory
          </Link>
          <nav className="flex gap-1">
            {TABS.map((tab) => (
              <NavLink
                key={tab.to}
                to={tab.to}
                className={({ isActive }) =>
                  `rounded-lg px-3 py-1.5 text-sm transition ${
                    isActive
                      ? "bg-accent/20 text-accent-soft"
                      : "text-slate-400 hover:bg-slate-700/40 hover:text-slate-200"
                  }`
                }
              >
                {tab.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6">{children}</main>
    </div>
  );
}
