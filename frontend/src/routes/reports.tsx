import { createFileRoute } from "@tanstack/react-router";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { CalendarDays, Download, FileSpreadsheet, Loader2, TrendingUp } from "lucide-react";
import {
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { PageHeader } from "@/components/page-header";
import { RiskBadge } from "@/components/status-badge";
import { EmptyState, ErrorState, LoadingCard, SectionCard, StatCard } from "@/components/stat-card";
import { Button } from "@/components/ui/button";
import { downloadReport, formatEventTime, getReport } from "@/lib/api";
import { RISK_LEVELS, type ReportRange, type RiskLevel } from "@/lib/types";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/reports")({
  head: () => ({
    meta: [
      { title: "Reports | SentinelOps" },
      { name: "description", content: "Aggregated safety reports, charts, and CSV / JSON exports." },
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

const RISK_COLORS: Record<RiskLevel, string> = {
  SAFE: "#16a34a",
  LOW: "#0284c7",
  MEDIUM: "#d97706",
  HIGH: "#ea580c",
  CRITICAL: "#dc2626",
};

function ReportsPage() {
  const [range, setRange] = useState<ReportRange>("24h");
  const [exporting, setExporting] = useState<"csv" | "json" | null>(null);

  const report = useQuery({ queryKey: ["report", range], queryFn: () => getReport(range) });

  const data = report.data;
  const totals = data?.totals;
  const byTypeEntries = Object.entries(data?.by_violation_type ?? {});
  const byDayEntries = Object.entries(data?.by_day ?? {});

  const pieData = RISK_LEVELS.map((level) => ({
    name: level,
    value: data?.by_risk[level] ?? 0,
    color: RISK_COLORS[level],
  })).filter((item) => item.value > 0);

  const barData = byTypeEntries.map(([type, count]) => ({
    name: type.replaceAll("_", " "),
    count,
  }));

  const timelineData = byDayEntries.map(([day, count]) => ({
    day: day.slice(5),
    events: count,
  }));

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
        title="Performance reports & analytics"
        description="Interactive visual analytics computed over recorded events, alerts, and safety violations."
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
            {/* Risk breakdown chart */}
            <SectionCard title="Risk distribution" description="Events categorized by risk level in range">
              {pieData.length === 0 ? (
                <EmptyState
                  icon={TrendingUp}
                  title="No events in this range"
                  description="Record events from the live monitor or by processing a video."
                  className="border-0 py-6"
                />
              ) : (
                <div className="flex flex-col items-center sm:flex-row sm:justify-around">
                  <div className="h-52 w-52 shrink-0">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={pieData}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          innerRadius={45}
                          outerRadius={75}
                          paddingAngle={3}
                        >
                          {pieData.map((entry) => (
                            <Cell key={entry.name} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "var(--color-card)",
                            borderColor: "var(--color-border)",
                            borderRadius: "0.5rem",
                            fontSize: "12px",
                          }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>

                  <ul className="mt-4 sm:mt-0 space-y-2 w-full max-w-xs">
                    {RISK_LEVELS.map((level) => {
                      const count = data.by_risk[level] ?? 0;
                      return (
                        <li key={level} className="flex items-center justify-between text-xs">
                          <span className="flex items-center gap-2">
                            <span
                              className="size-2.5 rounded-full"
                              style={{ backgroundColor: RISK_COLORS[level] }}
                            />
                            <span className="font-medium text-foreground">{level}</span>
                          </span>
                          <span className="font-display font-semibold">{count}</span>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}
            </SectionCard>

            {/* Violation types chart */}
            <SectionCard
              title="Violation types breakdown"
              description="Cumulative counts across recorded safety events"
            >
              {barData.length === 0 ? (
                <EmptyState
                  icon={TrendingUp}
                  title="No violations recorded"
                  description="Violation totals by type will appear here."
                  className="border-0 py-6"
                />
              ) : (
                <div className="h-56 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={barData} layout="vertical" margin={{ left: 20, right: 20, top: 10, bottom: 10 }}>
                      <XAxis type="number" tick={{ fontSize: 11 }} />
                      <YAxis dataKey="name" type="category" width={110} tick={{ fontSize: 11 }} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "var(--color-card)",
                          borderColor: "var(--color-border)",
                          borderRadius: "0.5rem",
                          fontSize: "12px",
                        }}
                      />
                      <Bar dataKey="count" fill="var(--color-primary)" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </SectionCard>
          </div>

          {/* Daily timeline */}
          <SectionCard title="Daily event timeline" description="Safety events recorded per day">
            {timelineData.length === 0 ? (
              <p className="text-xs text-muted-foreground">No daily data available in this range.</p>
            ) : (
              <div className="h-48 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={timelineData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <XAxis dataKey="day" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "var(--color-card)",
                        borderColor: "var(--color-border)",
                        borderRadius: "0.5rem",
                        fontSize: "12px",
                      }}
                    />
                    <Bar dataKey="events" fill="var(--color-primary)" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
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
