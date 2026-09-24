import { createFileRoute } from "@tanstack/react-router";
import { useCallback } from "react";
import {
  CameraOff,
  CircleStop,
  Cctv,
  Radar,
  ScanFace,
  ShieldAlert,
  ShieldCheck,
  Users,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader } from "@/components/page-header";
import { IdentityBadge, RiskBadge } from "@/components/status-badge";
import { EmptyState, SectionCard } from "@/components/stat-card";
import { useLiveCamera, type ConnectionState } from "@/hooks/use-live-camera";
import type { FaceIdentity, SafetyPayload } from "@/lib/types";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/monitor")({
  head: () => ({
    meta: [{ title: "Live Monitor | SentinelOps" }],
  }),
  component: MonitorPage,
});

const CONNECTION_LABELS: Record<ConnectionState, string> = {
  idle: "Idle",
  connecting: "Connecting…",
  connected: "Live",
  error: "Error",
  offline: "Disconnected",
};

const CONNECTION_TONES: Record<
  ConnectionState,
  "neutral" | "success" | "warning" | "danger" | "info"
> = {
  idle: "neutral",
  connecting: "info",
  connected: "success",
  error: "danger",
  offline: "danger",
};

function ConnectionBadge({ state }: { state: ConnectionState }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold ring-1 ring-inset",
        CONNECTION_TONES[state] === "success" && "bg-safe/10 text-safe ring-safe/30",
        CONNECTION_TONES[state] === "info" && "bg-primary/10 text-primary ring-primary/30",
        CONNECTION_TONES[state] === "danger" && "bg-critical/10 text-critical ring-critical/30",
        CONNECTION_TONES[state] === "warning" && "bg-medium/15 text-medium ring-medium/30",
        CONNECTION_TONES[state] === "neutral" && "bg-muted text-muted-foreground ring-border",
      )}
    >
      <span
        className={cn(
          "size-1.5 rounded-full",
          state === "connected"
            ? "bg-safe status-pulse"
            : state === "connecting"
              ? "bg-primary status-pulse"
              : state === "idle"
                ? "bg-muted-foreground"
                : "bg-critical",
        )}
      />
      {CONNECTION_LABELS[state]}
    </span>
  );
}

function MetricPill({
  label,
  value,
  tone,
}: {
  label: string;
  value: number | string;
  tone?: string | undefined;
}) {
  return (
    <div className="rounded-lg border bg-card px-3 py-2">
      <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
        {label}
      </p>
      <p className={cn("mt-1 font-display text-lg font-bold leading-none", tone)}>{value}</p>
    </div>
  );
}

function WorkerRow({ identity }: { identity: FaceIdentity }) {
  const detail = identity.verified
    ? `score ${(identity.score * 100).toFixed(0)}% · mean ${(identity.mean_score * 100).toFixed(0)}%`
    : (identity.error ??
      `${identity.matches.length} worker${identity.matches.length === 1 ? "" : "s"} checked`);

  return (
    <li className="flex items-center gap-3 rounded-lg border px-3 py-2.5">
      <span
        className={cn(
          "grid size-9 shrink-0 place-items-center rounded-full",
          identity.verified ? "bg-safe/12 text-safe" : "bg-muted text-muted-foreground",
        )}
      >
        {identity.verified ? (
          <ShieldCheck className="size-4.5" />
        ) : (
          <ScanFace className="size-4.5" />
        )}
      </span>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-semibold">
          {identity.verified ? identity.worker_name || identity.worker_id : "Unknown person"}
        </p>
        <p className="truncate text-[11px] text-muted-foreground">
          {identity.track_id != null ? `Track #${identity.track_id} · ` : ""}
          {detail}
        </p>
      </div>
      <IdentityBadge verified={identity.verified} />
    </li>
  );
}

function MonitorPage() {
  const { attachVideo, attachCanvasImage, connection, start, stop, latestPayload, fps, error } =
    useLiveCamera();

  const handleStart = useCallback(() => void start(), [start]);

  const summary: SafetyPayload["summary"] | null = latestPayload?.summary ?? null;
  const identities = latestPayload?.identities ?? [];
  const identitySummary = latestPayload?.identity_summary ?? null;
  const violations = latestPayload?.violations ?? [];
  const dangerZones = latestPayload?.danger_zones ?? [];
  const proximity = (latestPayload?.distance_results ?? []).filter(
    (item) => item.distance_m_est != null || item.distance_px != null,
  );
  const nearViolations = proximity.length;
  const fireCount = summary?.fire_count ?? 0;
  const smokeCount = summary?.smoke_count ?? 0;

  const riskLevel = latestPayload?.risk_level ?? "SAFE";

  return (
    <div>
      <PageHeader
        eyebrow="Live monitoring"
        title="Real-time AI safety pipeline"
        description="Webcam frames are streamed to the YOLO hazard model, ByteTrack tracking, and the Siamese face-verification backend. Results render live on the annotated feed."
        actions={
          <>
            {connection === "connected" ? (
              <Button variant="destructive" onClick={stop}>
                <CircleStop className="size-4" /> Stop camera
              </Button>
            ) : (
              <Button onClick={handleStart} disabled={connection === "connecting"}>
                <Cctv className="size-4" />
                {connection === "connecting" ? "Starting…" : "Start live camera"}
              </Button>
            )}
          </>
        }
      />

      {/* Status strip */}
      <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-6">
        <MetricPill label="Risk level" value={riskLevel} />
        <MetricPill label="Persons" value={summary?.person_count ?? 0} />
        <MetricPill
          label="Violations"
          value={summary?.violation_count ?? 0}
          tone={summary?.violation_count ? "text-high" : undefined}
        />
        <MetricPill
          label="Verified"
          value={identitySummary?.workers_verified ?? 0}
          tone="text-safe"
        />
        <MetricPill
          label="Unknown faces"
          value={identitySummary?.unknown_faces ?? 0}
          tone={identitySummary?.unknown_faces ? "text-medium" : undefined}
        />
        <MetricPill label="Processed FPS" value={fps} />
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
        {/* Annotated feed */}
        <Card className="overflow-hidden">
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="flex items-center gap-2 font-display text-sm font-semibold">
              <Radar className="size-4 text-primary" /> Annotated feed
            </CardTitle>
            <div className="flex items-center gap-2">
              <RiskBadge level={riskLevel} pulse={riskLevel === "CRITICAL"} />
              <ConnectionBadge state={connection} />
            </div>
          </CardHeader>
          <CardContent>
            <div className="relative aspect-video w-full overflow-hidden rounded-lg border bg-muted">
              {/* Raw webcam (shown while streaming so the operator sees themselves) */}
              <video
                ref={attachVideo}
                playsInline
                muted
                className={cn(
                  "absolute inset-0 h-full w-full object-cover",
                  connection === "connected" ? "opacity-100" : "opacity-0",
                )}
              />
              {/* Backend-annotated frame overlays the raw video once frames return */}
              <img
                ref={attachCanvasImage}
                alt="AI-annotated live camera feed"
                className={cn(
                  "absolute inset-0 h-full w-full object-contain transition-opacity duration-200",
                  connection === "connected" ? "opacity-100" : "opacity-0",
                )}
              />

              {connection !== "connected" ? (
                <div className="absolute inset-0 grid place-items-center bg-background/80 p-6 text-center">
                  <div>
                    <span className="mx-auto grid size-12 place-items-center rounded-full bg-muted">
                      <CameraOff className="size-5 text-muted-foreground" />
                    </span>
                    <p className="mt-3 text-sm font-semibold">
                      {connection === "connecting" ? "Connecting to camera…" : "Camera not active"}
                    </p>
                    <p className="mx-auto mt-1 max-w-sm text-xs leading-relaxed text-muted-foreground">
                      {error ??
                        "Start the camera to stream frames through the full AI pipeline: hazard detection, PPE, danger zones, and worker verification."}
                    </p>
                    {connection === "idle" || connection === "error" || connection === "offline" ? (
                      <Button size="sm" className="mt-4" onClick={handleStart}>
                        <Cctv className="size-4" /> Start live camera
                      </Button>
                    ) : null}
                  </div>
                </div>
              ) : null}

              {connection === "connected" ? (
                <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-1.5 rounded-md bg-foreground/75 px-2 py-1 text-[10px] font-semibold text-background backdrop-blur">
                  <span className="size-1.5 rounded-full bg-critical status-pulse" /> REC · {fps}{" "}
                  FPS
                </div>
              ) : null}
            </div>

            {/* Quick detections strip */}
            <div className="mt-3 flex flex-wrap gap-1.5">
              {(latestPayload?.detections ?? []).slice(0, 12).map((detection, index) => (
                <span
                  key={`${detection.class}-${index}`}
                  className="rounded-md border bg-muted px-2 py-0.5 text-[10.5px] font-medium"
                >
                  {detection.class}{" "}
                  <span className="text-muted-foreground">
                    {(detection.confidence * 100).toFixed(0)}%
                  </span>
                </span>
              ))}
              {(latestPayload?.detections ?? []).length === 0 ? (
                <span className="text-xs text-muted-foreground">No detections yet.</span>
              ) : null}
            </div>
          </CardContent>
        </Card>

        {/* Side panels */}
        <div className="space-y-4">
          <SectionCard
            title="Worker identification"
            description="Siamese face verification against registered workers"
          >
            {identities.length > 0 ? (
              <ul className="space-y-2">
                {identities.map((identity, index) => (
                  <WorkerRow
                    key={`${identity.track_id ?? identity.person_index}-${index}`}
                    identity={identity}
                  />
                ))}
              </ul>
            ) : (
              <EmptyState
                icon={ScanFace}
                title="No faces in view"
                description="When a person appears in the camera, their face is cropped and verified against every registered worker."
                className="border-0 py-6"
              />
            )}
            {identitySummary ? (
              <p className="mt-3 border-t pt-3 text-[11px] text-muted-foreground">
                {identitySummary.faces_seen} face{identitySummary.faces_seen === 1 ? "" : "s"} this
                frame · {identitySummary.workers_verified} verified ·{" "}
                {identitySummary.unknown_faces} unknown
              </p>
            ) : null}
          </SectionCard>

          <SectionCard
            title="PPE compliance"
            description="Anatomical hardhat / vest / mask matching"
          >
            {summary ? (
              <div className="space-y-2 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">PPE violations</span>
                  <span
                    className={cn(
                      "font-display font-bold",
                      summary.ppe_violations ? "text-high" : "text-safe",
                    )}
                  >
                    {summary.ppe_violations}
                  </span>
                </div>
                {Object.entries(summary.violation_counts)
                  .filter(([type]) => ["NO_HARDHAT", "NO_SAFETY_VEST", "NO_MASK"].includes(type))
                  .map(([type, count]) => (
                    <div
                      key={type}
                      className="flex items-center justify-between rounded-md bg-critical/5 px-2.5 py-1.5 text-xs"
                    >
                      <span className="font-semibold">{type.replaceAll("_", " ")}</span>
                      <span className="text-muted-foreground">×{count}</span>
                    </div>
                  ))}
                {summary.ppe_violations === 0 ? (
                  <p className="rounded-md bg-safe/8 px-2.5 py-1.5 text-xs text-safe">
                    All detected workers compliant.
                  </p>
                ) : null}
              </div>
            ) : (
              <p className="text-xs text-muted-foreground">Waiting for camera data…</p>
            )}
          </SectionCard>

          <SectionCard title="Active hazards" description="Zones, proximity, fire / smoke">
            {violations.length > 0 ? (
              <ul className="space-y-2">
                {violations.slice(0, 6).map((violation, index) => (
                  <li
                    key={`${violation.type}-${index}`}
                    className="flex items-start gap-2 rounded-md border px-2.5 py-2 text-xs"
                  >
                    <ShieldAlert className="mt-0.5 size-3.5 shrink-0 text-high" />
                    <div className="min-w-0">
                      <p className="font-semibold">{violation.message ?? violation.type}</p>
                      {violation.severity ? (
                        <p className="text-muted-foreground">
                          severity {violation.severity.toLowerCase()}
                        </p>
                      ) : null}
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="space-y-2 text-xs text-muted-foreground">
                <p>
                  Danger zones detected:{" "}
                  <span className="font-semibold text-foreground">{dangerZones.length}</span>
                </p>
                <p>
                  Proximity measurements:{" "}
                  <span className="font-semibold text-foreground">{nearViolations}</span>
                </p>
                <p>
                  Fire / smoke:{" "}
                  <span
                    className={cn(
                      "font-semibold",
                      fireCount || smokeCount ? "text-critical" : "text-safe",
                    )}
                  >
                    {fireCount} / {smokeCount}
                  </span>
                </p>
                {violations.length === 0 ? (
                  <p className="text-safe">No active violations on this frame.</p>
                ) : null}
              </div>
            )}
          </SectionCard>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 font-display text-sm font-semibold">
                <Users className="size-4 text-primary" /> Pipeline status
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-1.5 text-[11px] text-muted-foreground">
              <p>Hazard model: YOLO + ByteTrack tracking</p>
              <p>Fire / smoke model: temporal confirmation gate</p>
              <p>Face recognition: Siamese (siamesemodelv2.h5)</p>
              <p>Session state: {CONNECTION_LABELS[connection]}</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
