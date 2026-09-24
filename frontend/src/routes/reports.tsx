import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { CalendarDays, Download, FileSpreadsheet, Loader2, ShieldAlert, TrendingUp } from "lucide-react";
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
      { title: "Reports & Analytics | SentinelOps" },
      { name: "description", content: "Aggregated site safety reports, incident metrics, and CSV/JSON exports." },
    ],
  }),
  component: ReportsPage,
});

const RANGES: Array<{ key: ReportRange; label: string }> = [
  { key: "24h", label: "Last 24 Hours" },
  { key: "7d", label: "7 Days" },
  { key: "30d", label: "30 Days" },
  { key: "all", label: "All Time" },
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
      anchor.download = `sentinelops_safety_report_${range}.${format}`;
      anchor.click();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(null);
    }
  };

  return (
    <div>
      <PageHeader
        eyebrow="Safety Intelligence & Analytics"
        title="Performance Reports & Exports"
        description="Comprehensive aggregates computed by the safety engine over recorded event and alert history. Export detailed datasets for OSHA compliance or site safety audits."
        actions={
          <div className="flex gap-2">
            <Button
              variant="outline"
              disabled={exporting !== null}
              onClick={() => void handleDownload("csv")}
            >
              {exporting === "csv" ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <FileSpreadsheet className="size-4 text-primary" />
              )}
              Export CSV Dataset
            </Button>
            <Button
              variant="outline"
              disabled={exporting !== null}
              onClick={() => void handleDownload("json")}
            >
              {exporting === "json" ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Download className="size-4 text-primary" />
              )}
              Export JSON Report
            </Button>
          </div>
        }
      />

      {/* Range picker */}
      <div className="mb-4 flex flex-wrap items-center gap-3">
        <span className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground">
          <CalendarDays className="size-4" /> Reporting Window:
        </span>
        <div
          className="flex gap-1 rounded-lg border bg-card p-1 shadow-sm"
          role="tablist"
          aria-label="Report time window"
        >
          {RANGES.map((item) => (
            <button
              key={item.key}
              type="button"
              role="tab"
              aria-selected={range === item.key}
              onClick={() => setRange(item.key)}
              className={cn(
                "rounded-md px-3 py-1.5 text-xs font-semibold transition-all",
                range === item.key
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted/50",
              )}
            >
              {item.label}
            </button>
          ))}
        </div>
        {data ? (
          <span className="ml-auto text-xs text-muted-foreground font-medium">
            Generated at {formatEventTime(data.generated_at)}
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
          message="Could not load report data. Verify backend API server is running on port 8000."
          onRetry={() => void report.refetch()}
        />
      ) : !data ? null : (
        <div className="space-y-4">
          {/* Totals */}
          <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
            <StatCard label="Total Events" value={totals?.events ?? 0} detail="Recorded in reporting window" />
            <StatCard
              label="Safety Alerts"
              value={totals?.alerts ?? 0}
              detail={`${totals?.alerts_unacknowledged ?? 0} currently unacknowledged`}
              tone={(totals?.alerts_unacknowledged ?? 0) > 0 ? "warning" : "success"}
            />
            <StatCard
              label="Frames Evaluated"
              value={totals?.frames_processed ?? 0}
              detail="Total AI forward passes"
              tone="info"
            />
            <StatCard
              label="Evidence Captured"
              value={(totals?.events_with_image ?? 0) + (totals?.events_with_clip ?? 0)}
              detail={`${totals?.events_with_clip ?? 0} MP4 clips · ${totals?.events_with_image ?? 0} snapshots`}
              tone="info"
            />
          </div>

          <div className="grid gap-4 xl:grid-cols-2">
            {/* Risk breakdown */}
            <SectionCard title="Risk Level Breakdown" description="Distribution of events by risk severity level">
              {byDayEntries.length === 0 && (totals?.events ?? 0) === 0 ? (
                <EmptyState
                  icon={TrendingUp}
                  title="No events in this time frame"
                  description="Record events by running the webcam monitor, image inspection, or video detector."
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
                        <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-muted">
                          <div
                            className={cn(
                              "h-full rounded-full transition-all duration-300",
                              level === "SAFE" && "bg-safe",
                              level === "LOW" && "bg-low",
                              level === "MEDIUM" && "bg-medium",
                              level === "HIGH" && "bg-high",
                              level === "CRITICAL" && "bg-critical",
                            )}
                            style={{ width: `${total ? (count / total) * 100 : 0}%` }}
                          />
                        </div>
                        <span className="w-10 text-right font-display text-sm font-bold text-foreground">
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
              title="Violation Class Frequency"
              description="Cumulative occurrences by safety rule violation type"
            >
              {byType.length === 0 ? (
                <EmptyState
                  icon={TrendingUp}
                  title="No violations recorded"
                  description="Rule violation counts by class will accumulate here."
                  className="border-0 py-6"
                />
              ) : (
                <ul className="space-y-3">
                  {byType.map(([type, count]) => (
                    <li key={type}>
                      <div className="mb-1 flex justify-between text-xs font-medium">
                        <span className="text-foreground">{type.replaceAll("_", " ")}</span>
                        <span className="font-semibold text-foreground">{count}</span>
                      </div>
                      <div className="h-2 overflow-hidden rounded-full bg-muted">
                        <div
                          className="h-full rounded-full bg-primary transition-all duration-300"
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
          <SectionCard title="Daily Incident Timeline" description="Event volume trends over calendar days">
            {byDayEntries.length === 0 ? (
              <p className="text-xs text-muted-foreground">No events recorded in this window.</p>
            ) : (
              <div className="flex h-36 items-end gap-2 overflow-x-auto pb-2 pt-4">
                {byDayEntries.map(([day, count]) => {
                  const max = Math.max(...byDayEntries.map(([, value]) => value), 1);
                  return (
                    <div
                      key={day}
                      className="flex min-w-10 flex-1 flex-col items-center gap-1.5"
                      title={`${day}: ${count} event${count === 1 ? "" : "s"}`}
                    >
                      <span className="text-[10px] font-bold text-foreground">{count}</span>
                      <div
                        className="w-full rounded-t-md bg-primary/80 hover:bg-primary transition-colors"
                        style={{ height: `${Math.max((count / max) * 100, 6)}%` }}
                      />
                      <span className="text-[10px] font-medium text-muted-foreground">{day.slice(5)}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </SectionCard>

          {/* Most severe events */}
          <SectionCard
            title="Most Severe Safety Incidents"
            description="Top recorded events ranked by risk level and violation multiplicity"
          >
            {data.top_events.length === 0 ? (
              <EmptyState
                icon={ShieldAlert}
                title="No severe incidents recorded"
                description="High severity incidents will be highlighted in this section."
                className="border-0 py-6"
              />
            ) : (
              <ul className="divide-y">
                {data.top_events.slice(0, 8).map((event) => (
                  <li
                    key={event.id}
                    className="flex flex-wrap items-center gap-x-3 gap-y-1 py-3 text-sm"
                  >
                    <RiskBadge level={event.risk_level} size="sm" />
                    <span className="font-semibold text-foreground">{event.violation_count} violation{event.violation_count === 1 ? "" : "s"}</span>
                    <span className="text-xs text-muted-foreground">
                      {Object.keys(event.violation_counts).slice(0, 3).join(" · ") || "Multiple risk factors"}
                    </span>
                    <span className="ml-auto text-xs text-muted-foreground font-medium">
                      Source: <span className="capitalize text-foreground">{event.source}</span> · {formatEventTime(event.ts)}
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
