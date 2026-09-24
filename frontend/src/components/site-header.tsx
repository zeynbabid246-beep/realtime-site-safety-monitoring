import { useEffect, useState, type ReactNode } from "react";
import { Link, useRouterState } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  Bell,
  Construction,
  FileBarChart,
  History,
  Menu,
  Moon,
  ScanFace,
  Sun,
  TriangleAlert,
  Video,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { getHealth, listAlerts } from "@/lib/api";
import { useTheme } from "@/hooks/use-theme";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", icon: Activity, exact: true },
  { to: "/monitor", label: "Live Monitor", icon: Video },
  { to: "/workers", label: "Workers", icon: ScanFace },
  { to: "/history", label: "History", icon: History },
  { to: "/reports", label: "Reports", icon: FileBarChart },
] as const;

function NavLink({
  to,
  label,
  icon: Icon,
  exact,
  onNavigate,
}: {
  to: string;
  label: string;
  icon: typeof Activity;
  exact?: boolean;
  onNavigate?: (() => void) | undefined;
}) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const active = exact ? pathname === to : pathname.startsWith(to);
  return (
    <Link
      to={to}
      onClick={onNavigate}
      aria-current={active ? "page" : undefined}
      className={cn(
        "flex items-center gap-3 rounded-md px-3 py-2 text-[13px] font-medium transition-colors",
        active
          ? "bg-sidebar-primary/15 text-sidebar-primary ring-1 ring-inset ring-sidebar-primary/30"
          : "text-sidebar-foreground/65 hover:bg-sidebar-accent hover:text-sidebar-foreground",
      )}
    >
      <Icon className="size-4 shrink-0" />
      {label}
    </Link>
  );
}

function SidebarNav({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav aria-label="Primary" className="flex flex-1 flex-col gap-1 p-3">
      <p className="px-3 pb-2 pt-1 text-[10px] font-bold uppercase tracking-[0.16em] text-sidebar-foreground/35">
        Operations
      </p>
      {NAV_ITEMS.map((item) => (
        <NavLink key={item.to} {...item} onNavigate={onNavigate} />
      ))}
    </nav>
  );
}

export function SiteHeader({ children: page }: { children: ReactNode }) {
  const { theme, toggleTheme } = useTheme();
  const [mobileOpen, setMobileOpen] = useState(false);

  // Backend health + unacknowledged alert count (auto-refresh).
  const health = useQuery({
    queryKey: ["health"],
    queryFn: getHealth,
    retry: 1,
    refetchInterval: 30_000,
  });
  const alerts = useQuery({
    queryKey: ["alerts", "badge"],
    queryFn: () => listAlerts({ status: "new", limit: 100 }),
    refetchInterval: 15_000,
    retry: 1,
  });

  const backendOnline = health.isSuccess;
  const unacknowledged = alerts.data?.unacknowledged ?? 0;

  useEffect(() => {
    setMobileOpen(false);
  }, []);

  return (
    <div className="flex min-h-screen">
      {/* Desktop sidebar (dark accent rail) */}
      <aside className="sticky top-0 hidden h-screen w-56 shrink-0 flex-col border-r border-sidebar-border bg-sidebar lg:flex">
        <Link to="/" className="flex items-center gap-2.5 px-4 py-4">
          <span className="grid size-8 place-items-center rounded-lg bg-sidebar-primary/20 ring-1 ring-inset ring-sidebar-primary/40">
            <Construction className="size-4.5 text-sidebar-primary" />
          </span>
          <span className="font-display text-[15px] font-bold tracking-tight text-sidebar-foreground">
            SENTINEL<span className="text-sidebar-primary">OPS</span>
          </span>
        </Link>
        <SidebarNav />
        <div className="border-t border-sidebar-border p-3">
          <div className="flex items-center gap-2 rounded-md bg-sidebar-accent/60 px-3 py-2 text-[11px] text-sidebar-foreground/70">
            <span
              className={cn(
                "size-1.5 rounded-full",
                backendOnline ? "bg-safe status-pulse" : "bg-critical",
              )}
            />
            <span className="font-semibold">
              {backendOnline ? "Backend online" : "Backend offline"}
            </span>
          </div>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        {/* Top header */}
        <header className="sticky top-0 z-30 border-b bg-card/85 backdrop-blur-md">
          <div className="flex h-14 items-center gap-3 px-4 lg:px-6">
            <Button
              variant="ghost"
              size="icon"
              className="lg:hidden"
              aria-label={mobileOpen ? "Close navigation" : "Open navigation"}
              aria-expanded={mobileOpen}
              onClick={() => setMobileOpen((open) => !open)}
            >
              {mobileOpen ? <X className="size-5" /> : <Menu className="size-5" />}
            </Button>

            <div className="min-w-0 flex-1" />

            <div
              className={cn(
                "hidden items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold ring-1 ring-inset sm:flex",
                backendOnline
                  ? "bg-safe/10 text-safe ring-safe/30"
                  : "bg-critical/10 text-critical ring-critical/30",
              )}
            >
              <span
                className={cn(
                  "size-1.5 rounded-full",
                  backendOnline ? "bg-safe status-pulse" : "bg-critical status-pulse",
                )}
              />
              {backendOnline ? `API ${health.data?.version ?? ""}`.trim() : "API offline"}
            </div>

            <Link
              to="/history"
              aria-label={`Alerts${unacknowledged ? `, ${unacknowledged} unacknowledged` : ""}`}
              className="relative grid size-9 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <Bell className="size-4.5" />
              {unacknowledged > 0 ? (
                <span className="absolute -right-0.5 -top-0.5 grid min-w-4 place-items-center rounded-full bg-critical px-1 text-[9px] font-bold text-white">
                  {unacknowledged > 99 ? "99+" : unacknowledged}
                </span>
              ) : null}
            </Link>

            <Button
              variant="ghost"
              size="icon"
              aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
              onClick={toggleTheme}
            >
              {theme === "dark" ? <Sun className="size-4.5" /> : <Moon className="size-4.5" />}
            </Button>
          </div>
        </header>

        {/* Mobile drawer */}
        {mobileOpen ? (
          <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true">
            <div
              className="absolute inset-0 bg-foreground/40 backdrop-blur-sm"
              onClick={() => setMobileOpen(false)}
              aria-hidden="true"
            />
            <div className="absolute inset-y-0 left-0 flex w-60 flex-col bg-sidebar shadow-xl panel-rise">
              <div className="flex items-center justify-between px-4 py-4">
                <span className="font-display text-[15px] font-bold text-sidebar-foreground">
                  SENTINEL<span className="text-sidebar-primary">OPS</span>
                </span>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Close navigation"
                  className="text-sidebar-foreground/70"
                  onClick={() => setMobileOpen(false)}
                >
                  <X className="size-5" />
                </Button>
              </div>
              <SidebarNav onNavigate={() => setMobileOpen(false)} />
              <div className="border-t border-sidebar-border p-3">
                <div className="flex items-center gap-2 px-3 py-2 text-[11px] text-sidebar-foreground/70">
                  <TriangleAlert className="size-3.5" />
                  {unacknowledged} open alert{unacknowledged === 1 ? "" : "s"}
                </div>
              </div>
            </div>
          </div>
        ) : null}

        <main className="flex-1 px-4 py-4 lg:px-6 lg:py-5">{page}</main>
      </div>
    </div>
  );
}
