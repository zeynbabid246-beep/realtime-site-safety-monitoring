import { createFileRoute, Link } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import {
  ArrowRight,
  Cctv,
  Cone,
  Cpu,
  FileVideo,
  Flame,
  HardHat,
  ImageIcon,
  Radar,
  ScanFace,
  ShieldAlert,
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
          "AI-powered construction site safety monitoring platform: PPE, danger zones, proximity, fire, and worker verification.",
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
              className={cn("h-full transition-all duration-300", RISK_BG[level])}
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
  const faceOnline = workers.isSuccess;

  return (
    <div>
      <PageHeader
        eyebrow="Construction Safety AI Platform"
        title="Site Safety Operations Center"
        description="Live overview of real-time webcam detections, image & video inspections, PPE compliance, danger zone proximity, and worker identification."
        actions={
          <div className="flex gap-2">
            <Button asChild>
              <Link to="/monitor">
                <Cctv className="size-4" /> Open Detection Studio
              </Link>
            </Button>
          </div>
        }
      />

      {/* System status strip */}
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
          Safety Core Engine {backendOnline ? `v${health.data?.version ?? "?"} online` : "offline"}
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
            ? `Siamese Face Verifier Online · ${activeWorkers.length} active worker${activeWorkers.length === 1 ? "" : "s"}`
            : "Face recognition unavailable"}
        </span>
        <span className="inline-flex items-center gap-1.5 rounded-full bg-primary/10 text-primary ring-1 ring-inset ring-primary/30 px-2.5 py-1 font-semibold">
          <Radar className="size-3.5" /> YOLO Hazard & Fire Models Loaded
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
            detail={`${totals?.frames_processed ?? 0} total frames processed`}
            icon={ShieldCheck}
            tone="info"
          />
          <StatCard
            label="Active Alerts"
            value={totals?.alerts_unacknowledged ?? 0}
            detail={`${totals?.alerts ?? 0} alerts recorded total`}
            icon={Flame}
            tone={(totals?.alerts_unacknowledged ?? 0) > 0 ? "warning" : "success"}
          />
          <StatCard
            label="Critical Incidents"
            value={report24h.data?.by_risk.CRITICAL ?? 0}
            detail="Highest risk severity in 24h"
            icon={HardHat}
            tone={(report24h.data?.by_risk.CRITICAL ?? 0) > 0 ? "danger" : "success"}
          />
          <StatCard
            label="Registered Workers"
            value={activeWorkers.length}
            detail={`${(workers.data?.workers.length ?? 0) - activeWorkers.length} deactivated`}
            icon={Users}
            tone="success"
          />
        </div>
      )}

      {/* Studio Quick Launcher */}
      <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {[
          {
            to: "/monitor",
            label: "Live Camera Stream",
            detail: "Webcam live feed with real-time detection",
            icon: Cctv,
            color: "text-primary bg-primary/10",
          },
          {
            to: "/monitor",
            label: "Image Hazard Inspection",
            detail: "Upload single photo for full safety pass",
            icon: ImageIcon,
            color: "text-safe bg-safe/10",
          },
          {
            to: "/monitor",
            label: "Video Recording Processing",
            detail: "Batch video processing & annotated MP4",
            icon: FileVideo,
            color: "text-medium bg-medium/10",
          },
          {
            to: "/workers",
            label: "Worker Identity Portal",
            detail: "Manage identities & reference face images",
            icon: ScanFace,
            color: "text-info bg-info/10",
          },
        ].map((action, idx) => (
          <Link
            key={idx}
            to={action.to}
            className="group flex items-center gap-3 rounded-lg border bg-card p-4 transition-all hover:border-primary/40 hover:bg-accent/50 shadow-sm"
          >
            <span className={cn("grid size-10 shrink-0 place-items-center rounded-lg font-bold", action.color)}>
              <action.icon className="size-5" />
            </span>
            <span className="min-w-0 flex-1">
              <span className="block text-sm font-semibold text-foreground">{action.label}</span>
              <span className="block truncate text-xs text-muted-foreground">{action.detail}</span>
            </span>
            <ArrowRight className="size-4 text-muted-foreground transition-transform group-hover:translate-x-1" />
          </Link>
        ))}
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        {/* Risk mix */}
        <SectionCard
          title="Risk Distribution"
          description="Events by risk level over the last 24 hours"
          action={
            <Link to="/reports" className="text-xs font-semibold text-primary hover:underline">
              Full Analytics →
            </Link>
          }
        >
          {byRisk ? (
            <RiskMix byRisk={byRisk} />
          ) : (
            <p className="text-xs text-muted-foreground">Loading risk metrics…</p>
          )}
        </SectionCard>

        {/* Violation types */}
        <SectionCard title="Top Violations" description="24-hour totals by violation type">
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
                        className="h-full rounded-full bg-primary transition-all duration-300"
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

        {/* Worker Verification */}
        <SectionCard
          title="Worker Identification"
          description="Active registered identities for live verification"
          action={
            <Link to="/workers" className="text-xs font-semibold text-primary hover:underline">
              Manage Workers →
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
          title="Recent Detection Events"
          description="Latest frames that triggered recorded safety incidents"
          action={
            <Link to="/history" className="text-xs font-semibold text-primary hover:underline">
              View History & Alerts →
            </Link>
          }
        >
          {recentEvents.isLoading ? (
            <p className="text-xs text-muted-foreground">Loading recent events…</p>
          ) : recentEvents.isError ? (
            <ErrorState
              message="Could not load the event history."
              onRetry={() => void recentEvents.refetch()}
            />
          ) : events.length === 0 ? (
            <EmptyState
              icon={Cpu}
              title="No events recorded yet"
              description="Start the live camera or upload a video to record safety events."
            />
          ) : (
            <ul className="divide-y">
              {events.map((event) => (
                <li
                  key={event.id}
                  className="flex flex-wrap items-center gap-x-3 gap-y-1 py-2.5 text-sm"
                >
                  <RiskBadge level={event.risk_level} size="sm" />
                  <span className="font-semibold">
                    {event.violation_count} violation{event.violation_count === 1 ? "" : "s"}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {Object.keys(event.violation_counts).slice(0, 3).join(" · ") || "Standard inspection"}
                  </span>
                  <span
                    className="ml-auto text-xs text-muted-foreground"
                    title={formatEventTime(event.ts)}
                  >
                    Source: <span className="font-medium text-foreground capitalize">{event.source}</span> · {relativeTime(event.ts)}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </SectionCard>
      </div>
    </div>
  );
}
