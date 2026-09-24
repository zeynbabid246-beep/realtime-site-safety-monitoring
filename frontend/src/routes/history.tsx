import { createFileRoute, Link } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import {
  BellOff,
  CheckCheck,
  Cpu,
  FileImage,
  History as HistoryIcon,
  Search,
  Video,
} from "lucide-react";
import { PageHeader } from "@/components/page-header";
import { RISK_LABELS, RiskBadge, StatusBadge } from "@/components/status-badge";
import { EmptyState, ErrorState, LoadingCard, SectionCard, StatCard } from "@/components/stat-card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  acknowledgeAlert,
  evidenceUrl,
  formatEventTime,
  listAlerts,
  listEvents,
  relativeTime,
} from "@/lib/api";
import { RISK_LEVELS, type RiskLevel } from "@/lib/types";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/history")({
  head: () => ({
    meta: [
      { title: "History | SentinelOps" },
      {
        name: "description",
        content: "Detection history and safety incidents with risk, source, and status filters.",
      },
    ],
  }),
  component: HistoryPage,
});

type RiskFilter = "all" | RiskLevel;
type SourceFilter = "all" | "camera" | "video" | "image";

function HistoryPage() {
  const queryClient = useQueryClient();
  const [riskFilter, setRiskFilter] = useState<RiskFilter>("all");
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>("all");
  const [statusFilter, setStatusFilter] = useState<"all" | "new" | "acknowledged">("all");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const PAGE_SIZE = 15;

  const eventsQuery = useQuery({
    queryKey: ["events", "history"],
    queryFn: () => listEvents({ limit: 500 }),
    refetchInterval: 20_000,
  });

  const alertsQuery = useQuery({
    queryKey: ["alerts", statusFilter],
    queryFn: () =>
      listAlerts(statusFilter === "all" ? { limit: 100 } : { limit: 100, status: statusFilter }),
    refetchInterval: 20_000,
  });

  const ackMutation = useMutation({
    mutationFn: acknowledgeAlert,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["alerts"] });
      void queryClient.invalidateQueries({ queryKey: ["report"] });
    },
  });

  const events = useMemo(() => eventsQuery.data?.events ?? [], [eventsQuery.data]);

  const filteredEvents = useMemo(
    () =>
      events
        .filter((event) => (riskFilter === "all" ? true : event.risk_level === riskFilter))
        .filter((event) => (sourceFilter === "all" ? true : event.source === sourceFilter))
        .filter((event) => {
          const needle = search.trim().toLowerCase();
          if (!needle) return true;
          return (
            event.violation_count.toString().includes(needle) ||
            Object.keys(event.violation_counts).some((type) =>
              type.toLowerCase().includes(needle),
            ) ||
            event.source.toLowerCase().includes(needle)
          );
        }),
    [events, riskFilter, sourceFilter, search],
  );

  const pageCount = Math.max(1, Math.ceil(filteredEvents.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const pagedEvents = filteredEvents.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE);

  const alerts = alertsQuery.data?.alerts ?? [];
  const unacknowledged = alertsQuery.data?.unacknowledged ?? 0;

  const criticalCount = filteredEvents.filter((event) => event.risk_level === "CRITICAL").length;
  const ppeCount = filteredEvents.reduce(
    (sum, event) =>
      sum +
      (event.violation_counts["NO_HARDHAT"] ?? 0) +
      (event.violation_counts["NO_SAFETY_VEST"] ?? 0) +
      (event.violation_counts["NO_MASK"] ?? 0),
    0,
  );

  return (
    <div>
      <PageHeader
        eyebrow="Monitoring history"
        title="Events & incidents"
        description="Every recorded safety event and alert, with filters by risk level, source, and acknowledgement status."
      />

      {/* Summary */}
      <div className="mb-4 grid grid-cols-2 gap-3 xl:grid-cols-4">
        <StatCard
          label="Events loaded"
          value={filteredEvents.length}
          detail={`${events.length} total in store`}
          icon={HistoryIcon}
        />
        <StatCard
          label="Critical events"
          value={criticalCount}
          detail="In current filter"
          icon={Cpu}
          tone={criticalCount ? "danger" : "success"}
        />
        <StatCard
          label="PPE violations"
          value={ppeCount}
          detail="Cumulative in view"
          icon={Video}
          tone={ppeCount ? "warning" : "success"}
        />
        <StatCard
          label="Open alerts"
          value={unacknowledged}
          detail="Awaiting acknowledgement"
          icon={BellOff}
          tone={unacknowledged ? "warning" : "success"}
        />
      </div>

      {/* Alerts feed */}
      <SectionCard
        title="Safety alerts"
        description="HIGH / CRITICAL events pushed to the alert channel"
        className="mb-4"
        action={
          <div className="flex gap-1 rounded-md border p-0.5">
            {(["all", "new", "acknowledged"] as const).map((status) => (
              <button
                key={status}
                type="button"
                onClick={() => setStatusFilter(status)}
                className={cn(
                  "rounded px-2.5 py-1 text-[11px] font-semibold capitalize transition-colors",
                  statusFilter === status
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {status}
              </button>
            ))}
          </div>
        }
      >
        {alertsQuery.isLoading ? (
          <p className="text-xs text-muted-foreground">Loading alerts…</p>
        ) : alertsQuery.isError ? (
          <ErrorState message="Could not load alerts." onRetry={() => void alertsQuery.refetch()} />
        ) : alerts.length === 0 ? (
          <EmptyState
            icon={CheckCheck}
            title="No alerts"
            description="HIGH and CRITICAL safety events will appear here as the pipeline detects them."
            className="border-0 py-6"
          />
        ) : (
          <ul className="space-y-2">
            {alerts.slice(0, 8).map((alert) => (
              <li
                key={alert.id}
                className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg border px-3 py-2.5"
              >
                <RiskBadge level={alert.level as RiskLevel} size="sm" />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-semibold">{alert.title}</p>
                  <p className="truncate text-xs text-muted-foreground">{alert.message}</p>
                </div>
                <span
                  className="text-[11px] text-muted-foreground"
                  title={formatEventTime(alert.ts)}
                >
                  {relativeTime(alert.ts)}
                </span>
                {alert.status === "new" ? (
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={ackMutation.isPending}
                    onClick={() => ackMutation.mutate(alert.id)}
                  >
                    <CheckCheck className="size-3.5" /> Acknowledge
                  </Button>
                ) : (
                  <StatusBadge tone="success">Acknowledged</StatusBadge>
                )}
              </li>
            ))}
          </ul>
        )}
      </SectionCard>

      {/* Filters */}
      <SectionCard
        title="Event log"
        description="Frames that produced recorded safety events"
        action={
          <div className="relative w-44 sm:w-64">
            <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={search}
              onChange={(event) => {
                setSearch(event.target.value);
                setPage(0);
              }}
              placeholder="Search violations…"
              className="h-8 pl-9 text-xs"
              aria-label="Search events"
            />
          </div>
        }
      >
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold text-muted-foreground">Risk:</span>
          {(["all", ...RISK_LEVELS] as const).map((level) => (
            <button
              key={level}
              type="button"
              onClick={() => {
                setRiskFilter(level as RiskFilter);
                setPage(0);
              }}
              className={cn(
                "rounded-full px-2.5 py-1 text-[11px] font-semibold ring-1 ring-inset transition-colors",
                riskFilter === level
                  ? "bg-primary text-primary-foreground ring-primary"
                  : "text-muted-foreground ring-border hover:text-foreground",
              )}
            >
              {level === "all" ? "All" : RISK_LABELS[level as RiskLevel]}
            </button>
          ))}
          <span className="ml-3 text-xs font-semibold text-muted-foreground">Source:</span>
          {(["all", "camera", "video", "image"] as const).map((source) => (
            <button
              key={source}
              type="button"
              onClick={() => {
                setSourceFilter(source);
                setPage(0);
              }}
              className={cn(
                "rounded-full px-2.5 py-1 text-[11px] font-semibold capitalize ring-1 ring-inset transition-colors",
                sourceFilter === source
                  ? "bg-primary text-primary-foreground ring-primary"
                  : "text-muted-foreground ring-border hover:text-foreground",
              )}
            >
              {source}
            </button>
          ))}
        </div>

        {eventsQuery.isLoading ? (
          <LoadingCard rows={6} />
        ) : eventsQuery.isError ? (
          <ErrorState
            message="Could not load the event log."
            onRetry={() => void eventsQuery.refetch()}
          />
        ) : pagedEvents.length === 0 ? (
          <EmptyState
            icon={HistoryIcon}
            title="No events match the filters"
            description="Try clearing the search or widening the risk/source filters. Events appear here once the pipeline records them."
          />
        ) : (
          <>
            <div className="overflow-x-auto rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-28">Time</TableHead>
                    <TableHead className="w-24">Risk</TableHead>
                    <TableHead className="w-20">Source</TableHead>
                    <TableHead className="w-16 text-right">Persons</TableHead>
                    <TableHead>Violations</TableHead>
                    <TableHead className="w-20 text-right">Evidence</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {pagedEvents.map((event) => (
                    <TableRow key={event.id}>
                      <TableCell className="whitespace-nowrap text-xs">
                        {formatEventTime(event.ts)}
                      </TableCell>
                      <TableCell>
                        <RiskBadge level={event.risk_level} size="sm" />
                      </TableCell>
                      <TableCell className="text-xs capitalize">{event.source}</TableCell>
                      <TableCell className="text-right text-xs">{event.persons}</TableCell>
                      <TableCell>
                        <div className="flex flex-wrap gap-1">
                          {Object.entries(event.violation_counts)
                            .slice(0, 4)
                            .map(([type, count]) => (
                              <span
                                key={type}
                                className="rounded bg-muted px-1.5 py-0.5 text-[10px] font-medium"
                              >
                                {type.replaceAll("_", " ")} ×{count}
                              </span>
                            ))}
                          {Object.keys(event.violation_counts).length === 0 ? (
                            <span className="text-xs text-muted-foreground">—</span>
                          ) : null}
                        </div>
                      </TableCell>
                      <TableCell className="text-right">
                        {event.evidence_image ? (
                          <a
                            href={evidenceUrl(event.evidence_image) ?? "#"}
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center gap-1 text-xs font-semibold text-primary hover:underline"
                          >
                            <FileImage className="size-3.5" /> View
                          </a>
                        ) : (
                          <span className="text-xs text-muted-foreground">—</span>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>

            {/* Pagination */}
            {pageCount > 1 ? (
              <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
                <span>
                  {safePage * PAGE_SIZE + 1}–
                  {Math.min((safePage + 1) * PAGE_SIZE, filteredEvents.length)} of{" "}
                  {filteredEvents.length}
                </span>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={safePage === 0}
                    onClick={() => setPage(safePage - 1)}
                  >
                    Previous
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={safePage >= pageCount - 1}
                    onClick={() => setPage(safePage + 1)}
                  >
                    Next
                  </Button>
                </div>
              </div>
            ) : null}
          </>
        )}
      </SectionCard>

      <p className="mt-4 text-xs text-muted-foreground">
        Need aggregates and exports? See the{" "}
        <Link to="/reports" className="font-semibold text-primary hover:underline">
          Reports
        </Link>{" "}
        page.
      </p>
    </div>
  );
}
