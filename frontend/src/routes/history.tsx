import { createFileRoute, Link } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import {
  AlertCircle,
  Bell,
  BellOff,
  CheckCheck,
  Cpu,
  Download,
  FileImage,
  Film,
  History as HistoryIcon,
  Image as ImageIcon,
  Play,
  Search,
  ShieldAlert,
  Video,
  X,
} from "lucide-react";
import { PageHeader } from "@/components/page-header";
import { RISK_LABELS, RiskBadge, StatusBadge } from "@/components/status-badge";
import { EmptyState, ErrorState, LoadingCard, SectionCard, StatCard } from "@/components/stat-card";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
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
  acknowledgeAllAlerts,
  evidenceUrl,
  formatEventTime,
  listAlerts,
  listEvents,
  listEvidence,
  relativeTime,
} from "@/lib/api";
import { RISK_LEVELS, type RiskLevel, type SafetyEvent } from "@/lib/types";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/history")({
  head: () => ({
    meta: [
      { title: "History & Evidence | SentinelOps" },
      {
        name: "description",
        content: "Detection event history, safety alerts feed, and captured evidence gallery.",
      },
    ],
  }),
  component: HistoryPage,
});

type HistoryTab = "events" | "alerts" | "evidence";
type RiskFilter = "all" | RiskLevel;
type SourceFilter = "all" | "camera" | "video" | "image";

interface LightboxMedia {
  title: string;
  type: "image" | "video";
  url: string;
  ts: number;
  riskLevel: RiskLevel;
  violationCounts: Record<string, number>;
  source: string;
}

function HistoryPage() {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState<HistoryTab>("events");
  const [riskFilter, setRiskFilter] = useState<RiskFilter>("all");
  const [sourceFilter, setSourceFilter] = useState<SourceFilter>("all");
  const [statusFilter, setStatusFilter] = useState<"all" | "new" | "acknowledged">("all");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const [lightboxMedia, setLightboxMedia] = useState<LightboxMedia | null>(null);
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

  const evidenceQuery = useQuery({
    queryKey: ["evidence", "gallery"],
    queryFn: () => listEvidence(60),
    refetchInterval: 20_000,
  });

  const ackMutation = useMutation({
    mutationFn: acknowledgeAlert,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["alerts"] });
      void queryClient.invalidateQueries({ queryKey: ["report"] });
    },
  });

  const ackAllMutation = useMutation({
    mutationFn: acknowledgeAllAlerts,
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
  const evidenceItems = evidenceQuery.data?.events ?? [];

  const criticalCount = filteredEvents.filter((event) => event.risk_level === "CRITICAL").length;
  const ppeCount = filteredEvents.reduce(
    (sum, event) =>
      sum +
      (event.violation_counts["NO_HARDHAT"] ?? 0) +
      (event.violation_counts["NO_SAFETY_VEST"] ?? 0) +
      (event.violation_counts["NO_MASK"] ?? 0),
    0,
  );

  const openEventImage = (event: SafetyEvent) => {
    if (!event.evidence_image) return;
    const url = evidenceUrl(event.evidence_image);
    if (!url) return;
    setLightboxMedia({
      title: `Event #${event.id} Snapshot`,
      type: "image",
      url,
      ts: event.ts,
      riskLevel: event.risk_level,
      violationCounts: event.violation_counts,
      source: event.source,
    });
  };

  const openEventClip = (event: SafetyEvent) => {
    if (!event.evidence_clip) return;
    const url = evidenceUrl(event.evidence_clip);
    if (!url) return;
    setLightboxMedia({
      title: `Event #${event.id} Video Clip`,
      type: "video",
      url,
      ts: event.ts,
      riskLevel: event.risk_level,
      violationCounts: event.violation_counts,
      source: event.source,
    });
  };

  return (
    <div>
      <PageHeader
        eyebrow="Monitoring history & evidence"
        title="Events & evidence center"
        description="Review all recorded safety incidents, process pending alerts, and inspect captured high-resolution evidence snapshots and clips."
      />

      {/* KPI Cards */}
      <div className="mb-4 grid grid-cols-2 gap-3 xl:grid-cols-4">
        <StatCard
          label="Events logged"
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

      {/* View Switcher Tabs */}
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2 border-b pb-3">
        <div className="flex items-center gap-1.5 bg-muted p-1 rounded-lg">
          <button
            type="button"
            onClick={() => setActiveTab("events")}
            className={cn(
              "flex items-center gap-2 rounded-md px-3.5 py-1.5 text-xs font-semibold transition-colors",
              activeTab === "events"
                ? "bg-card text-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            <HistoryIcon className="size-3.5" /> Event Log ({filteredEvents.length})
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("alerts")}
            className={cn(
              "flex items-center gap-2 rounded-md px-3.5 py-1.5 text-xs font-semibold transition-colors",
              activeTab === "alerts"
                ? "bg-card text-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            <Bell className="size-3.5" /> Active Alerts
            {unacknowledged > 0 ? (
              <span className="rounded-full bg-critical px-1.5 py-0.2 text-[10px] font-bold text-white">
                {unacknowledged}
              </span>
            ) : null}
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("evidence")}
            className={cn(
              "flex items-center gap-2 rounded-md px-3.5 py-1.5 text-xs font-semibold transition-colors",
              activeTab === "evidence"
                ? "bg-card text-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            <ImageIcon className="size-3.5" /> Evidence Gallery ({evidenceItems.length})
          </button>
        </div>

        {activeTab === "alerts" && unacknowledged > 0 ? (
          <Button
            size="sm"
            disabled={ackAllMutation.isPending}
            onClick={() => ackAllMutation.mutate()}
          >
            <CheckCheck className="size-3.5" /> Acknowledge All Alerts
          </Button>
        ) : null}
      </div>

      {/* TAB 1: EVENT LOG */}
      {activeTab === "events" && (
        <SectionCard
          title="Safety event log"
          description="Frames that produced recorded safety incidents"
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
              description="Try clearing search or widening filters. Events appear here as detected by the pipeline."
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
                      <TableHead className="w-32 text-right">Evidence</TableHead>
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
                        <TableCell className="text-right whitespace-nowrap">
                          <div className="flex items-center justify-end gap-2">
                            {event.evidence_image ? (
                              <button
                                type="button"
                                onClick={() => openEventImage(event)}
                                className="inline-flex items-center gap-1 text-xs font-semibold text-primary hover:underline"
                              >
                                <FileImage className="size-3.5" /> Snapshot
                              </button>
                            ) : null}
                            {event.evidence_clip ? (
                              <button
                                type="button"
                                onClick={() => openEventClip(event)}
                                className="inline-flex items-center gap-1 text-xs font-semibold text-primary hover:underline"
                              >
                                <Film className="size-3.5" /> Clip
                              </button>
                            ) : null}
                            {!event.evidence_image && !event.evidence_clip ? (
                              <span className="text-xs text-muted-foreground">—</span>
                            ) : null}
                          </div>
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
      )}

      {/* TAB 2: ALERTS FEED */}
      {activeTab === "alerts" && (
        <SectionCard
          title="Safety alert channel"
          description="Pushed HIGH and CRITICAL safety incidents requiring review"
          action={
            <div className="flex items-center gap-2">
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
              title="No alerts found"
              description="HIGH and CRITICAL safety events will appear here as detected."
              className="border-0 py-8"
            />
          ) : (
            <ul className="space-y-2.5">
              {alerts.map((alert) => (
                <li
                  key={alert.id}
                  className="flex flex-wrap items-center gap-x-3 gap-y-1.5 rounded-lg border p-3.5 bg-card"
                >
                  <RiskBadge level={alert.level as RiskLevel} size="sm" />
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-semibold">{alert.title}</p>
                    <p className="text-xs text-muted-foreground mt-0.5">{alert.message}</p>
                  </div>
                  <span
                    className="text-[11px] text-muted-foreground whitespace-nowrap"
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
      )}

      {/* TAB 3: EVIDENCE GALLERY */}
      {activeTab === "evidence" && (
        <SectionCard
          title="Captured evidence media"
          description="Annotated snapshots and video clips captured during high-risk safety events"
        >
          {evidenceQuery.isLoading ? (
            <LoadingCard rows={4} />
          ) : evidenceQuery.isError ? (
            <ErrorState message="Could not load evidence gallery." onRetry={() => void evidenceQuery.refetch()} />
          ) : evidenceItems.length === 0 ? (
            <EmptyState
              icon={ImageIcon}
              title="No evidence captured yet"
              description="Evidence images and clips are automatically recorded on HIGH and CRITICAL risk events."
              className="border-0 py-8"
            />
          ) : (
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {evidenceItems.map((item) => {
                const imgUrl = evidenceUrl(item.evidence_image);
                const clipUrl = evidenceUrl(item.evidence_clip);
                return (
                  <div key={item.id} className="group flex flex-col overflow-hidden rounded-xl border bg-card shadow-xs transition-colors hover:border-primary/50">
                    <div className="relative aspect-video w-full bg-black overflow-hidden">
                      {imgUrl ? (
                        <img
                          src={imgUrl}
                          alt={`Evidence event ${item.id}`}
                          className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-105"
                        />
                      ) : clipUrl ? (
                        <div className="grid h-full w-full place-items-center bg-muted">
                          <Play className="size-8 text-primary" />
                        </div>
                      ) : null}

                      <div className="absolute left-2 top-2 flex items-center gap-1.5">
                        <RiskBadge level={item.risk_level} size="sm" />
                      </div>

                      {clipUrl ? (
                        <span className="absolute right-2 top-2 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-bold text-white backdrop-blur">
                          MP4 Clip
                        </span>
                      ) : null}
                    </div>

                    <div className="flex flex-1 flex-col p-3 text-xs">
                      <div className="flex items-center justify-between">
                        <span className="font-semibold text-foreground">Event #{item.id}</span>
                        <span className="text-muted-foreground">{relativeTime(item.ts)}</span>
                      </div>
                      <div className="mt-1 flex flex-wrap gap-1">
                        {Object.entries(item.violation_counts).slice(0, 2).map(([type, cnt]) => (
                          <span key={type} className="rounded bg-muted px-1.5 py-0.5 text-[10px]">
                            {type.replaceAll("_", " ")} ×{cnt}
                          </span>
                        ))}
                      </div>

                      <div className="mt-auto pt-3 flex gap-2 border-t mt-2">
                        {imgUrl ? (
                          <Button
                            size="sm"
                            variant="outline"
                            className="flex-1 h-7 text-[11px]"
                            onClick={() =>
                              setLightboxMedia({
                                title: `Event #${item.id} Snapshot`,
                                type: "image",
                                url: imgUrl,
                                ts: item.ts,
                                riskLevel: item.risk_level,
                                violationCounts: item.violation_counts,
                                source: item.source,
                              })
                            }
                          >
                            <FileImage className="size-3" /> View image
                          </Button>
                        ) : null}
                        {clipUrl ? (
                          <Button
                            size="sm"
                            variant="outline"
                            className="flex-1 h-7 text-[11px]"
                            onClick={() =>
                              setLightboxMedia({
                                title: `Event #${item.id} Clip`,
                                type: "video",
                                url: clipUrl,
                                ts: item.ts,
                                riskLevel: item.risk_level,
                                violationCounts: item.violation_counts,
                                source: item.source,
                              })
                            }
                          >
                            <Film className="size-3" /> Play clip
                          </Button>
                        ) : null}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </SectionCard>
      )}

      {/* Lightbox Media Modal */}
      <Dialog open={lightboxMedia !== null} onOpenChange={(open) => !open && setLightboxMedia(null)}>
        <DialogContent className="sm:max-w-3xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 font-display text-sm font-semibold">
              <ShieldAlert className="size-4 text-primary" /> {lightboxMedia?.title}
            </DialogTitle>
          </DialogHeader>
          {lightboxMedia ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between gap-2 rounded-lg bg-muted px-3 py-2 text-xs">
                <div className="flex items-center gap-2">
                  <RiskBadge level={lightboxMedia.riskLevel} />
                  <span className="font-semibold text-foreground">Source: {lightboxMedia.source}</span>
                </div>
                <span className="text-muted-foreground">{formatEventTime(lightboxMedia.ts)}</span>
              </div>

              <div className="overflow-hidden rounded-lg border bg-black shadow">
                {lightboxMedia.type === "image" ? (
                  <img src={lightboxMedia.url} alt={lightboxMedia.title} className="w-full max-h-[500px] object-contain mx-auto" />
                ) : (
                  <video src={lightboxMedia.url} controls autoPlay className="w-full max-h-[500px] object-contain mx-auto" />
                )}
              </div>

              <div className="flex items-center justify-between border-t pt-2">
                <div className="flex flex-wrap gap-1">
                  {Object.entries(lightboxMedia.violationCounts).map(([type, cnt]) => (
                    <span key={type} className="rounded bg-muted px-2 py-0.5 text-xs font-medium">
                      {type.replaceAll("_", " ")} ×{cnt}
                    </span>
                  ))}
                </div>
                <a
                  href={lightboxMedia.url}
                  download
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1 rounded bg-primary px-3 py-1.5 text-xs font-semibold text-primary-foreground hover:bg-primary/90"
                >
                  <Download className="size-3.5" /> Download Media
                </a>
              </div>
            </div>
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  );
}
