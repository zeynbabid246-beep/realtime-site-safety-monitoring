import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowRight,
  Cone,
  Cpu,
  Flame,
  HardHat,
  ScanFace,
  ShieldCheck,
  Users,
  Video,
} from "lucide-react";
import { PageHeader } from "@/components/page-header";
import { RiskBadge } from "@/components/status-badge";
import {
  EmptyState,
  ErrorState,
  LoadingStatCards,
  SectionCard,
  StatCard,
} from "@/components/stat-card";
import { Button } from "@/components/ui/button";
import {
  formatEventTime,
  getHealth,
  getReport,
  listEvents,
  listWorkers,
  relativeTime,
} from "@/lib/api";
import { RISK_LEVELS, type RiskLevel } from "@/lib/types";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Dashboard | SentinelOps" },
      {
        name: "description",
        content:
          "AI-powered construction safety monitoring: PPE, zones, proximity, fire, and worker verification.",
      },
    ],
  }),
  component: DashboardPage,
});

const RISK_BG: Record<RiskLevel, string> = {
  SAFE: "bg-safe",
  LOW: "bg-low",
  MEDIUM: "bg-medium",
  HIGH: "bg-high",
  CRITICAL: "bg-critical",
};

function RiskMix({ byRisk }: { byRisk: Record<RiskLevel, number> }) {
  const total = RISK_LEVELS.reduce((sum, level) => sum + (byRisk[level] ?? 0), 0);

  return (
    <div>
      <div
        className="flex h-2.5 w-full overflow-hidden rounded-full bg-muted"
        role="img"
        aria-label="Risk level distribution"
      >
        {RISK_LEVELS.map((level) => {
          const value = byRisk[level] ?? 0;
          if (!value) return null;
          return (
            <div
              key={level}
              className={cn("h-full", RISK_BG[level])}
              style={{ width: `${(value / Math.max(total, 1)) * 100}%` }}
              title={`${level}: ${value}`}
            />
          );
        })}
      </div>
      <ul className="mt-4 space-y-2">
        {RISK_LEVELS.map((level) => (
          <li key={level} className="flex items-center justify-between text-sm">
            <span className="flex items-center gap-2">
              <span className={cn("size-2 rounded-full", RISK_BG[level])} />
              <span className="text-muted-foreground">{level}</span>
            </span>
            <span className="font-display font-semibold">{byRisk[level] ?? 0}</span>
          </li>
        ))}
      </ul>
      {total === 0 ? (
        <p className="mt-3 text-xs text-muted-foreground">No events recorded in this period yet.</p>
      ) : null}
    </div>
  );
}

function DashboardPage() {
  const health = useQuery({ queryKey: ["health"], queryFn: getHealth, retry: 1 });
  const report24h = useQuery({
    queryKey: ["report", "24h"],
    queryFn: () => getReport("24h"),
    refetchInterval: 30_000,
  });
  const recentEvents = useQuery({
    queryKey: ["events", "recent"],
    queryFn: () => listEvents({ limit: 8 }),
    refetchInterval: 15_000,
  });
  const workers = useQuery({ queryKey: ["workers"], queryFn: listWorkers, retry: 1 });

  const totals = report24h.data?.totals;
  const byRisk = report24h.data?.by_risk;
  const byType = report24h.data?.by_violation_type ?? {};
  const topViolationTypes = Object.entries(byType).slice(0, 5);
  const activeWorkers = workers.data?.workers.filter((worker) => worker.active) ?? [];
  const events = recentEvents.data?.events ?? [];

  const backendOnline = health.isSuccess;
  // /face/workers answers 200 only when the recognition layer is loaded;
  // 503 (face unavailable) surfaces here as an error state.
  const faceOnline = workers.isSuccess;

  return (
    <div>
      <PageHeader
        eyebrow="Construction safety platform"
        title="Site safety dashboard"
        description="Live overview of detections, PPE compliance, worker verification, and incidents across the monitoring pipeline."
        actions={
          <Button asChild>
            <Link to="/monitor">
              <Video className="size-4" /> Open live monitor
            </Link>
          </Button>
        }
      />

      {/* System status */}
      <div className="mb-4 flex flex-wrap items-center gap-2 text-xs">
        <span
          className={cn(
            "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 font-semibold ring-1 ring-inset",
            backendOnline
              ? "bg-safe/10 text-safe ring-safe/30"
              : "bg-critical/10 text-critical ring-critical/30",
          )}
        >
          <span
            className={cn(
              "size-1.5 rounded-full",
              backendOnline ? "bg-safe status-pulse" : "bg-critical",
            )}
          />
          Safety API {backendOnline ? `v${health.data?.version ?? "?"} online` : "offline"}
        </span>
        <span
          className={cn(
            "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 font-semibold ring-1 ring-inset",
            faceOnline
              ? "bg-safe/10 text-safe ring-safe/30"
              : "bg-medium/15 text-medium ring-medium/30",
          )}
        >
          <ScanFace className="size-3.5" />
          {faceOnline
            ? `Face recognition online · ${activeWorkers.length} worker${activeWorkers.length === 1 ? "" : "s"}`
            : "Face recognition unavailable"}
        </span>
      </div>

      {/* KPI row */}
      {report24h.isLoading ? (
        <LoadingStatCards />
      ) : report24h.isError ? (
        <ErrorState
          message="Could not load statistics from the backend. Verify the API server is running on port 8000."
          onRetry={() => void report24h.refetch()}
        />
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <StatCard
            label="Events · 24h"
            value={totals?.events ?? 0}
            detail={`${totals?.frames_processed ?? 0} frames processed`}
            icon={ShieldCheck}
            tone="info"
          />
          <StatCard
            label="Active alerts"
            value={totals?.alerts_unacknowledged ?? 0}
            detail={`${totals?.alerts ?? 0} alerts total`}
            icon={Flame}
            tone={(totals?.alerts_unacknowledged ?? 0) > 0 ? "warning" : "success"}
          />
          <StatCard
            label="Critical events"
            value={report24h.data?.by_risk.CRITICAL ?? 0}
            detail="Highest severity in period"
            icon={HardHat}
            tone={(report24h.data?.by_risk.CRITICAL ?? 0) > 0 ? "danger" : "success"}
          />
          <StatCard
            label="Registered workers"
            value={activeWorkers.length}
            detail={`${(workers.data?.workers.length ?? 0) - activeWorkers.length} inactive`}
            icon={Users}
            tone="success"
          />
        </div>
      )}

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        {/* Risk mix */}
        <SectionCard
          title="Risk distribution"
          description="Events by risk level over the last 24 hours"
          action={
            <Link to="/reports" className="text-xs font-semibold text-primary hover:underline">
              Reports →
            </Link>
          }
        >
          {byRisk ? (
            <RiskMix byRisk={byRisk} />
          ) : (
            <p className="text-xs text-muted-foreground">Loading…</p>
          )}
        </SectionCard>

        {/* Violation types */}
        <SectionCard title="Top violations" description="24-hour totals by violation type">
          {topViolationTypes.length > 0 ? (
            <ul className="space-y-3">
              {topViolationTypes.map(([type, count]) => {
                const max = topViolationTypes[0]?.[1] ?? 1;
                return (
                  <li key={type}>
                    <div className="mb-1 flex justify-between text-xs">
                      <span className="font-medium">{type.replaceAll("_", " ")}</span>
                      <span className="font-semibold">{count}</span>
                    </div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                      <div
                        className="h-full rounded-full bg-primary"
                        style={{ width: `${(count / Math.max(max, 1)) * 100}%` }}
                      />
                    </div>
                  </li>
                );
              })}
            </ul>
          ) : (
            <EmptyState
              icon={Cone}
              title="No violations recorded"
              description="Nothing was flagged by the safety engine in this window."
              className="border-0 py-4"
            />
          )}
        </SectionCard>

        {/* Face recognition status */}
        <SectionCard
          title="Worker verification"
          description="Registered identities for live recognition"
          action={
            <Link to="/workers" className="text-xs font-semibold text-primary hover:underline">
              Manage →
            </Link>
          }
        >
          {workers.isLoading ? (
            <p className="text-xs text-muted-foreground">Loading workers…</p>
          ) : workers.isError ? (
            <div className="rounded-md bg-medium/10 px-3 py-2 text-xs leading-relaxed text-medium">
              Face recognition backend unavailable (TensorFlow or model missing on the server).
              Safety monitoring continues without identity verification.
            </div>
          ) : activeWorkers.length > 0 ? (
            <ul className="space-y-2">
              {activeWorkers.slice(0, 4).map((worker) => (
                <li key={worker.worker_id} className="flex items-center gap-2.5 text-sm">
                  <span className="grid size-8 shrink-0 place-items-center rounded-full bg-primary/10 text-primary">
                    <ScanFace className="size-4" />
                  </span>
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-semibold">{worker.name || worker.worker_id}</p>
                    <p className="truncate text-[11px] text-muted-foreground">
                      {worker.role || "Worker"} · {worker.reference_images.length} reference face
                      {worker.reference_images.length === 1 ? "" : "s"}
                    </p>
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState
              icon={ScanFace}
              title="No workers registered"
              description="Register workers to see verified identities on the live monitor."
              className="border-0 py-4"
            />
          )}
        </SectionCard>
      </div>

      {/* Recent events */}
      <div className="mt-4">
        <SectionCard
          title="Recent events"
          description="Latest frames that produced recorded safety events"
          action={
            <Link to="/history" className="text-xs font-semibold text-primary hover:underline">
              Full history →
            </Link>
          }
        >
          {recentEvents.isLoading ? (
            <p className="text-xs text-muted-foreground">Loading events…</p>
          ) : recentEvents.isError ? (
            <ErrorState
              message="Could not load the event history."
              onRetry={() => void recentEvents.refetch()}
            />
          ) : events.length === 0 ? (
            <EmptyState
              icon={Cpu}
              title="No events yet"
              description="Start the live monitor — events are recorded whenever the engine flags a risk above LOW."
            />
          ) : (
            <ul className="divide-y">
              {events.map((event) => (
                <li
                  key={event.id}
                  className="flex flex-wrap items-center gap-x-3 gap-y-1 py-2.5 text-sm"
                >
                  <RiskBadge level={event.risk_level} size="sm" />
                  <span className="font-medium">
                    {event.violation_count} violation{event.violation_count === 1 ? "" : "s"}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {Object.keys(event.violation_counts).slice(0, 3).join(" · ") || "—"}
                  </span>
                  <span
                    className="ml-auto text-xs text-muted-foreground"
                    title={formatEventTime(event.ts)}
                  >
                    {event.source} · {relativeTime(event.ts)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </SectionCard>
      </div>

      {/* Quick actions */}
      <div className="mt-4 grid gap-3 sm:grid-cols-3">
        {[
          {
            to: "/monitor",
            label: "Start monitoring",
            detail: "Webcam through the full AI pipeline",
            icon: Video,
          },
          {
            to: "/workers",
            label: "Register a worker",
            detail: "Faces used for live verification",
            icon: ScanFace,
          },
          {
            to: "/history",
            label: "Review incidents",
            detail: "Filter events and acknowledge alerts",
            icon: ArrowRight,
          },
        ].map((action) => (
          <Link
            key={action.to}
            to={action.to}
            className="group flex items-center gap-3 rounded-lg border bg-card p-4 transition-colors hover:border-primary/40 hover:bg-accent"
          >
            <span className="grid size-9 place-items-center rounded-lg bg-primary/10 text-primary">
              <action.icon className="size-4.5" />
            </span>
            <span className="min-w-0 flex-1">
              <span className="block text-sm font-semibold">{action.label}</span>
              <span className="block truncate text-xs text-muted-foreground">{action.detail}</span>
            </span>
            <ArrowRight className="size-4 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
          </Link>
        ))}
      </div>
    </div>
  );
}
