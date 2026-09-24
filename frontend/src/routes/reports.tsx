import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { CalendarDays, Download, FileSpreadsheet, Loader2, TrendingUp } from "lucide-react";
import { PageHeader } from "@/components/page-header";
import { RiskBadge } from "@/components/status-badge";
import { EmptyState, ErrorState, LoadingCard, SectionCard, StatCard } from "@/components/stat-card";
import { Button } from "@/components/ui/button";
import { downloadReport, getReport, formatEventTime } from "@/lib/api";
import { RISK_LEVELS, type ReportRange } from "@/lib/types";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/reports")({
  head: () => ({
    meta: [
      { title: "Reports | SentinelOps" },
      { name: "description", content: "Aggregated safety reports and CSV / JSON exports." },
    ],
  }),
  component: ReportsPage,
});

const RANGES: Array<{ key: ReportRange; label: string }> = [
  { key: "24h", label: "24 hours" },
  { key: "7d", label: "7 days" },
  { key: "30d", label: "30 days" },
  { key: "all", label: "All time" },
];

function ReportsPage() {
  const [range, setRange] = useState<ReportRange>("24h");
  const [exporting, setExporting] = useState<"csv" | "json" | null>(null);

  const report = useQuery({ queryKey: ["report", range], queryFn: () => getReport(range) });

  const data = report.data;
  const totals = data?.totals;
  const byType = Object.entries(data?.by_violation_type ?? {});
  const maxTypeCount = byType[0]?.[1] ?? 0;
  const byDayEntries = Object.entries(data?.by_day ?? {});

  const handleDownload = async (format: "csv" | "json") => {
    setExporting(format);
    try {
      const response = await downloadReport(range, format);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `safety_report_${range}.${format}`;
      anchor.click();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(null);
    }
  };

  return (
    <div>
      <PageHeader
        eyebrow="Safety intelligence"
        title="Performance reports"
        description="Aggregates computed by the backend over the recorded event and alert history."
        actions={
          <>
            <Button
              variant="outline"
              disabled={exporting !== null}
              onClick={() => void handleDownload("csv")}
            >
              {exporting === "csv" ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <FileSpreadsheet className="size-4" />
              )}
              Export CSV
            </Button>
            <Button
              variant="outline"
              disabled={exporting !== null}
              onClick={() => void handleDownload("json")}
            >
              {exporting === "json" ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Download className="size-4" />
              )}
              Export JSON
            </Button>
          </>
        }
      />

      {/* Range picker */}
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <CalendarDays className="size-4 text-muted-foreground" />
        <div
          className="flex gap-1 rounded-md border bg-card p-0.5"
          role="tablist"
          aria-label="Report range"
        >
          {RANGES.map((item) => (
            <button
              key={item.key}
              type="button"
              role="tab"
              aria-selected={range === item.key}
              onClick={() => setRange(item.key)}
              className={cn(
                "rounded px-3 py-1.5 text-xs font-semibold transition-colors",
                range === item.key
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {item.label}
            </button>
          ))}
        </div>
        {data ? (
          <span className="text-xs text-muted-foreground">
            generated {formatEventTime(data.generated_at)}
          </span>
        ) : null}
      </div>

      {report.isLoading ? (
        <div className="space-y-4">
          <LoadingCard rows={2} />
          <LoadingCard rows={4} />
        </div>
      ) : report.isError ? (
        <ErrorState
          message="Could not load the report. Make sure the safety API is running and has recorded events."
          onRetry={() => void report.refetch()}
        />
      ) : !data ? null : (
        <div className="space-y-4">
          {/* Totals */}
          <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
            <StatCard label="Events" value={totals?.events ?? 0} detail="Recorded in range" />
            <StatCard
              label="Alerts"
              value={totals?.alerts ?? 0}
              detail={`${totals?.alerts_unacknowledged ?? 0} unacknowledged`}
              tone={(totals?.alerts_unacknowledged ?? 0) > 0 ? "warning" : "success"}
            />
            <StatCard
              label="Frames processed"
              value={totals?.frames_processed ?? 0}
              detail="All-time counter"
              tone="info"
            />
            <StatCard
              label="Evidence captured"
              value={(totals?.events_with_image ?? 0) + (totals?.events_with_clip ?? 0)}
              detail={`${totals?.events_with_clip ?? 0} clips · ${totals?.events_with_image ?? 0} snapshots`}
              tone="info"
            />
          </div>

          <div className="grid gap-4 xl:grid-cols-2">
            {/* Risk breakdown */}
            <SectionCard title="Risk breakdown" description="Events by risk level in range">
              {byDayEntries.length === 0 && (totals?.events ?? 0) === 0 ? (
                <EmptyState
                  icon={TrendingUp}
                  title="No events in this range"
                  description="Record events from the live monitor or by processing a video."
                  className="border-0 py-6"
                />
              ) : (
                <ul className="space-y-2.5">
                  {RISK_LEVELS.map((level) => {
                    const count = data.by_risk[level] ?? 0;
                    const total = RISK_LEVELS.reduce(
                      (sum, key) => sum + (data.by_risk[key] ?? 0),
                      0,
                    );
                    return (
                      <li key={level} className="flex items-center gap-3">
                        <RiskBadge level={level} size="sm" className="w-24 justify-center" />
                        <div className="h-2 flex-1 overflow-hidden rounded-full bg-muted">
                          <div
                            className={cn(
                              "h-full rounded-full",
                              level === "SAFE" && "bg-safe",
                              level === "LOW" && "bg-low",
                              level === "MEDIUM" && "bg-medium",
                              level === "HIGH" && "bg-high",
                              level === "CRITICAL" && "bg-critical",
                            )}
                            style={{ width: `${total ? (count / total) * 100 : 0}%` }}
                          />
                        </div>
                        <span className="w-10 text-right font-display text-sm font-semibold">
                          {count}
                        </span>
                      </li>
                    );
                  })}
                </ul>
              )}
            </SectionCard>

            {/* Violation types */}
            <SectionCard
              title="Violation types"
              description="Cumulative counts across recorded events"
            >
              {byType.length === 0 ? (
                <EmptyState
                  icon={TrendingUp}
                  title="No violations recorded"
                  description="Violation totals by type will appear here."
                  className="border-0 py-6"
                />
              ) : (
                <ul className="space-y-3">
                  {byType.map(([type, count]) => (
                    <li key={type}>
                      <div className="mb-1 flex justify-between text-xs">
                        <span className="font-medium">{type.replaceAll("_", " ")}</span>
                        <span className="font-semibold">{count}</span>
                      </div>
                      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
                        <div
                          className="h-full rounded-full bg-primary"
                          style={{ width: `${maxTypeCount ? (count / maxTypeCount) * 100 : 0}%` }}
                        />
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </SectionCard>
          </div>

          {/* Daily timeline */}
          <SectionCard title="Daily timeline" description="Events per day">
            {byDayEntries.length === 0 ? (
              <p className="text-xs text-muted-foreground">No data in this range.</p>
            ) : (
              <div className="flex h-32 items-end gap-1.5 overflow-x-auto pb-1">
                {byDayEntries.map(([day, count]) => {
                  const max = Math.max(...byDayEntries.map(([, value]) => value), 1);
                  return (
                    <div
                      key={day}
                      className="flex min-w-8 flex-1 flex-col items-center gap-1"
                      title={`${day}: ${count}`}
                    >
                      <div
                        className="w-full rounded-t bg-primary/80"
                        style={{ height: `${Math.max((count / max) * 100, 4)}%` }}
                      />
                      <span className="text-[9px] text-muted-foreground">{day.slice(5)}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </SectionCard>

          {/* Most severe events */}
          <SectionCard
            title="Most severe events"
            description="Top events ordered by risk level and violation count"
          >
            {data.top_events.length === 0 ? (
              <EmptyState
                icon={TrendingUp}
                title="Nothing recorded yet"
                description="Severe events will be listed here."
                className="border-0 py-6"
              />
            ) : (
              <ul className="divide-y">
                {data.top_events.slice(0, 8).map((event) => (
                  <li
                    key={event.id}
                    className="flex flex-wrap items-center gap-x-3 gap-y-1 py-2.5 text-sm"
                  >
                    <RiskBadge level={event.risk_level} size="sm" />
                    <span className="font-medium">{event.violation_count} violations</span>
                    <span className="text-xs text-muted-foreground">
                      {Object.keys(event.violation_counts).slice(0, 3).join(" · ") || "—"}
                    </span>
                    <span className="ml-auto text-xs text-muted-foreground">
                      {event.source} · {formatEventTime(event.ts)}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </SectionCard>
        </div>
      )}
    </div>
  );
}
