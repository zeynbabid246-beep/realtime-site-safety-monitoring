import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  CameraOff,
  Cctv,
  CheckCircle2,
  CircleStop,
  FileImage,
  FileVideo,
  Flame,
  HardHat,
  Image as ImageIcon,
  Loader2,
  Play,
  Radar,
  ScanFace,
  ShieldAlert,
  ShieldCheck,
  UploadCloud,
  Users,
  Video,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { PageHeader } from "@/components/page-header";
import { IdentityBadge, RiskBadge, StatusBadge } from "@/components/status-badge";
import { EmptyState, SectionCard } from "@/components/stat-card";
import { useLiveCamera, type ConnectionState } from "@/hooks/use-live-camera";
import {
  detectImage,
  getApiBase,
  listEvidence,
  outputUrl,
  evidenceUrl,
  formatEventTime,
  relativeTime,
} from "@/lib/api";
import type {
  FaceIdentity,
  ImageDetectResponse,
  RiskLevel,
  SafetyPayload,
} from "@/lib/types";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/monitor")({
  head: () => ({
    meta: [{ title: "Safety Detection Studio | SentinelOps" }],
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
  const [activeTab, setActiveTab] = useState<"webcam" | "image" | "video" | "evidence">("webcam");

  /* Live webcam hook */
  const {
    attachVideo,
    attachCanvasImage,
    connection,
    start,
    stop,
    latestPayload,
    fps,
    error,
    hasFrame,
  } = useLiveCamera();

  const handleStart = useCallback(() => void start(), [start]);

  /* Image Upload State */
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);

  const imageMutation = useMutation({
    mutationFn: (file: File) => detectImage(file),
  });

  const handleImageSelect = (file: File) => {
    setImageFile(file);
    setImagePreviewUrl(URL.createObjectURL(file));
    imageMutation.mutate(file);
  };

  /* Video Upload State */
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [isUploadingVideo, setIsUploadingVideo] = useState(false);
  const [videoResultUrl, setVideoResultUrl] = useState<string | null>(null);
  const [videoMetadata, setVideoMetadata] = useState<{
    framesProcessed: string | null;
    failedFrames: string | null;
    maxRisk: string | null;
  } | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);

  const handleVideoUpload = async (file: File) => {
    setVideoFile(file);
    setIsUploadingVideo(true);
    setVideoError(null);
    setVideoResultUrl(null);
    setVideoMetadata(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${getApiBase()}/safety/detect/video`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Video processing failed: ${response.statusText}`);
      }

      const framesProcessed = response.headers.get("X-Frames-Processed");
      const failedFrames = response.headers.get("X-Frames-Failed");
      const maxRisk = response.headers.get("X-Max-Risk-Level");

      setVideoMetadata({ framesProcessed, failedFrames, maxRisk });

      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      setVideoResultUrl(objectUrl);
    } catch (err: unknown) {
      setVideoError(err instanceof Error ? err.message : "Failed to process video");
    } finally {
      setIsUploadingVideo(false);
    }
  };

  /* Evidence Query */
  const evidenceQuery = useQuery({
    queryKey: ["evidence", "gallery"],
    queryFn: () => listEvidence(60),
    refetchInterval: 15_000,
  });

  const summary = latestPayload?.summary ?? null;
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
        eyebrow="Safety Detection Studio"
        title="Multi-source AI Monitoring Platform"
        description="Stream real-time webcam video, upload high-resolution site images, or process full site video recordings through our end-to-end hazard, PPE, proximity, and worker identification models."
      />

      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as typeof activeTab)} className="mt-2">
        <TabsList className="grid w-full grid-cols-4 max-w-2xl mb-4">
          <TabsTrigger value="webcam" className="flex items-center gap-2">
            <Cctv className="size-4" /> Live Camera
          </TabsTrigger>
          <TabsTrigger value="image" className="flex items-center gap-2">
            <ImageIcon className="size-4" /> Image Detection
          </TabsTrigger>
          <TabsTrigger value="video" className="flex items-center gap-2">
            <FileVideo className="size-4" /> Video Processing
          </TabsTrigger>
          <TabsTrigger value="evidence" className="flex items-center gap-2">
            <Radar className="size-4" /> Evidence Gallery
          </TabsTrigger>
        </TabsList>

        {/* ========================================================= */}
        {/* WEBCAM MODE */}
        {/* ========================================================= */}
        <TabsContent value="webcam" className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="font-semibold text-muted-foreground">Stream controls:</span>
              <ConnectionBadge state={connection} />
            </div>
            {connection === "connected" ? (
              <Button variant="destructive" size="sm" onClick={stop}>
                <CircleStop className="size-4" /> Stop camera
              </Button>
            ) : (
              <Button size="sm" onClick={handleStart} disabled={connection === "connecting"}>
                <Cctv className="size-4" />
                {connection === "connecting" ? "Starting…" : "Start live camera"}
              </Button>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-6">
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
            <Card className="overflow-hidden">
              <CardHeader className="flex-row items-center justify-between space-y-0">
                <CardTitle className="flex items-center gap-2 font-display text-sm font-semibold">
                  <Radar className="size-4 text-primary" /> Annotated Feed
                </CardTitle>
                <div className="flex items-center gap-2">
                  <RiskBadge level={riskLevel} pulse={riskLevel === "CRITICAL"} />
                  <ConnectionBadge state={connection} />
                </div>
              </CardHeader>
              <CardContent>
                <div className="relative aspect-[4/3] w-full overflow-hidden rounded-lg border bg-muted">
                  <video
                    ref={attachVideo}
                    playsInline
                    muted
                    className={cn(
                      "absolute inset-0 h-full w-full object-cover",
                      connection === "connected" && !hasFrame ? "opacity-100" : "opacity-0",
                    )}
                  />
                  <img
                    ref={attachCanvasImage}
                    alt="AI-annotated live camera feed"
                    className={cn(
                      "absolute inset-0 h-full w-full object-contain transition-opacity duration-200",
                      connection === "connected" && hasFrame ? "opacity-100" : "opacity-0",
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
                      <span className="size-1.5 rounded-full bg-critical status-pulse" /> REC · {fps} FPS
                    </div>
                  ) : null}
                </div>

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

            <div className="space-y-4">
              <SectionCard
                title="Worker Identification"
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
                    description="When a person appears in the camera, their face is cropped and verified against registered workers."
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
                title="PPE Compliance"
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

              <SectionCard title="Active Hazards" description="Zones, proximity, fire / smoke">
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
            </div>
          </div>
        </TabsContent>

        {/* ========================================================= */}
        {/* IMAGE DETECTION MODE */}
        {/* ========================================================= */}
        <TabsContent value="image" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="font-display text-base">Image Hazard Analysis</CardTitle>
              <CardDescription>
                Upload a site photo to run hazard detection, PPE evaluation, danger zone geometry, and worker face recognition on a single image.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <label className="flex flex-col items-center justify-center w-full h-40 border-2 border-dashed rounded-lg cursor-pointer bg-muted/40 hover:bg-muted/70 transition-colors">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <UploadCloud className="w-10 h-10 mb-2 text-primary" />
                  <p className="mb-1 text-sm font-semibold">Click to upload or drag & drop</p>
                  <p className="text-xs text-muted-foreground">PNG, JPG, or WEBP (max 10MB)</p>
                </div>
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleImageSelect(file);
                  }}
                  aria-label="Upload site image"
                />
              </label>

              {imageMutation.isPending && (
                <div className="flex items-center justify-center py-8 gap-3 text-sm font-medium text-primary">
                  <Loader2 className="size-5 animate-spin" /> Processing image through safety pipeline…
                </div>
              )}

              {imageMutation.isError && (
                <div className="p-4 rounded-lg bg-critical/10 border border-critical/20 text-xs text-critical">
                  Failed to analyze image: {(imageMutation.error as Error).message}
                </div>
              )}

              {imageMutation.isSuccess && imageMutation.data && (
                <div className="space-y-6 mt-4">
                  <div className="grid gap-4 md:grid-cols-2">
                    {/* Original Image */}
                    <div className="space-y-2">
                      <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                        Original Photo
                      </p>
                      {imagePreviewUrl && (
                        <div className="overflow-hidden rounded-lg border bg-black aspect-[4/3] flex items-center justify-center">
                          <img
                            src={imagePreviewUrl}
                            alt="Original site upload"
                            className="max-h-full max-w-full object-contain"
                          />
                        </div>
                      )}
                    </div>

                    {/* Annotated Output */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                          Annotated AI Output
                        </p>
                        {imageMutation.data.result?.risk_level && (
                          <RiskBadge level={imageMutation.data.result.risk_level} />
                        )}
                      </div>
                      <div className="overflow-hidden rounded-lg border bg-black aspect-[4/3] flex items-center justify-center">
                        <img
                          src={outputUrl(imageMutation.data.annotated_image_path) ?? ""}
                          alt="Annotated detection output"
                          className="max-h-full max-w-full object-contain"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Summary Cards */}
                  {imageMutation.data.result && (
                    <div className="grid gap-4 md:grid-cols-3">
                      <SectionCard title="Detected Objects" description="YOLO detections in scene">
                        <div className="flex flex-wrap gap-1.5">
                          {imageMutation.data.result.detections.map((det, idx) => (
                            <span key={idx} className="rounded border bg-muted px-2 py-0.5 text-xs font-medium">
                              {det.class} ({Math.round(det.confidence * 100)}%)
                            </span>
                          ))}
                          {imageMutation.data.result.detections.length === 0 && (
                            <p className="text-xs text-muted-foreground">No objects detected.</p>
                          )}
                        </div>
                      </SectionCard>

                      <SectionCard title="Violations & Risks" description="Rule matches & severe risks">
                        <ul className="space-y-2">
                          {imageMutation.data.result.violations.map((v, idx) => (
                            <li key={idx} className="flex items-start gap-2 text-xs">
                              <ShieldAlert className="size-4 text-high shrink-0 mt-0.5" />
                              <div>
                                <p className="font-semibold">{v.message ?? v.type}</p>
                                {v.severity && <p className="text-muted-foreground">Severity: {v.severity}</p>}
                              </div>
                            </li>
                          ))}
                          {imageMutation.data.result.violations.length === 0 && (
                            <p className="text-xs text-safe flex items-center gap-1.5">
                              <CheckCircle2 className="size-4" /> No violations detected in image.
                            </p>
                          )}
                        </ul>
                      </SectionCard>

                      <SectionCard title="Identified Workers" description="Face recognition matches">
                        <ul className="space-y-2">
                          {imageMutation.data.result.identities.map((id, idx) => (
                            <WorkerRow key={idx} identity={id} />
                          ))}
                          {imageMutation.data.result.identities.length === 0 && (
                            <p className="text-xs text-muted-foreground">No worker faces identified.</p>
                          )}
                        </ul>
                      </SectionCard>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* ========================================================= */}
        {/* VIDEO PROCESSING MODE */}
        {/* ========================================================= */}
        <TabsContent value="video" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="font-display text-base">Video File Safety Inspection</CardTitle>
              <CardDescription>
                Upload site surveillance footage or mobile recorded video. The system processes frame-by-frame with tracking, danger zone evaluation, temporal fire confirmation, and evidence generation.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <label className="flex flex-col items-center justify-center w-full h-40 border-2 border-dashed rounded-lg cursor-pointer bg-muted/40 hover:bg-muted/70 transition-colors">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <FileVideo className="w-10 h-10 mb-2 text-primary" />
                  <p className="mb-1 text-sm font-semibold">Select video or drag & drop</p>
                  <p className="text-xs text-muted-foreground">MP4, AVI, or MOV format</p>
                </div>
                <input
                  type="file"
                  accept="video/mp4,video/avi,video/quicktime,video/x-matroska"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) void handleVideoUpload(file);
                  }}
                  aria-label="Upload site video"
                />
              </label>

              {isUploadingVideo && (
                <div className="flex flex-col items-center justify-center py-10 space-y-3">
                  <Loader2 className="size-8 animate-spin text-primary" />
                  <p className="text-sm font-medium">Processing video through safety engine…</p>
                  <p className="text-xs text-muted-foreground">This may take a minute depending on video length.</p>
                </div>
              )}

              {videoError && (
                <div className="p-4 rounded-lg bg-critical/10 border border-critical/20 text-xs text-critical">
                  Video processing failed: {videoError}
                </div>
              )}

              {videoResultUrl && (
                <div className="space-y-4 mt-4">
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b pb-3">
                    <div>
                      <h4 className="font-semibold text-sm">Processed Video Output</h4>
                      <p className="text-xs text-muted-foreground">{videoFile?.name}</p>
                    </div>
                    {videoMetadata?.maxRisk && (
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-medium text-muted-foreground">Max Risk Detected:</span>
                        <RiskBadge level={videoMetadata.maxRisk as RiskLevel} />
                      </div>
                    )}
                  </div>

                  {videoMetadata && (
                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                      <MetricPill label="Processed Frames" value={videoMetadata.framesProcessed ?? "—"} />
                      <MetricPill label="Failed Frames" value={videoMetadata.failedFrames ?? "0"} />
                      <MetricPill label="Highest Severity" value={videoMetadata.maxRisk ?? "SAFE"} />
                    </div>
                  )}

                  <div className="overflow-hidden rounded-lg border bg-black aspect-video max-w-4xl mx-auto">
                    <video controls src={videoResultUrl} className="w-full h-full">
                      Your browser does not support the video tag.
                    </video>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* ========================================================= */}
        {/* EVIDENCE GALLERY MODE */}
        {/* ========================================================= */}
        <TabsContent value="evidence" className="space-y-4">
          <SectionCard
            title="Captured Evidence Snapshots & Clips"
            description="Automatic snapshots and MP4 clips recorded when HIGH or CRITICAL hazards are detected"
          >
            {evidenceQuery.isLoading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="size-6 animate-spin text-primary" />
              </div>
            ) : evidenceQuery.isError ? (
              <p className="text-xs text-destructive">Failed to load evidence gallery.</p>
            ) : (evidenceQuery.data?.events ?? []).length === 0 ? (
              <EmptyState
                icon={FileImage}
                title="No evidence captured yet"
                description="Snapshots and MP4 clips will appear here whenever a HIGH or CRITICAL hazard event triggers."
              />
            ) : (
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {evidenceQuery.data?.events.map((item) => (
                  <Card key={item.id} className="overflow-hidden border">
                    <div className="relative aspect-video bg-black flex items-center justify-center">
                      {item.evidence_image ? (
                        <img
                          src={evidenceUrl(item.evidence_image) ?? ""}
                          alt={`Evidence #${item.id}`}
                          className="w-full h-full object-cover"
                        />
                      ) : item.evidence_clip ? (
                        <video
                          src={evidenceUrl(item.evidence_clip) ?? ""}
                          controls
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <FileImage className="size-8 text-muted-foreground" />
                      )}
                      <div className="absolute top-2 right-2">
                        <RiskBadge level={item.risk_level} size="sm" />
                      </div>
                    </div>
                    <CardContent className="p-3 text-xs space-y-1.5">
                      <div className="flex items-center justify-between text-muted-foreground">
                        <span className="font-medium text-foreground">Source: {item.source}</span>
                        <span>{relativeTime(item.ts)}</span>
                      </div>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {Object.entries(item.violation_counts).map(([type, count]) => (
                          <span key={type} className="rounded bg-muted px-1.5 py-0.5 text-[10px] font-medium">
                            {type.replaceAll("_", " ")} ×{count}
                          </span>
                        ))}
                      </div>
                      {item.evidence_clip && (
                        <a
                          href={evidenceUrl(item.evidence_clip) ?? "#"}
                          target="_blank"
                          rel="noreferrer"
                          className="inline-flex items-center gap-1 text-[11px] font-semibold text-primary hover:underline mt-2"
                        >
                          <Play className="size-3" /> Watch MP4 Clip
                        </a>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </SectionCard>
        </TabsContent>
      </Tabs>
    </div>
  );
}
