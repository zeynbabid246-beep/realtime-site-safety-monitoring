import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useEffect, useRef, useState, type ChangeEvent } from "react";
import {
  AlertTriangle,
  Camera,
  CameraOff,
  Cctv,
  CircleStop,
  Download,
  FileImage,
  FileVideo,
  Flame,
  HardHat,
  Loader2,
  Radar,
  RotateCcw,
  ScanFace,
  ShieldAlert,
  ShieldCheck,
  Truck,
  UploadCloud,
  Volume2,
  VolumeX,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader } from "@/components/page-header";
import { IdentityBadge, RiskBadge } from "@/components/status-badge";
import { EmptyState, SectionCard } from "@/components/stat-card";
import { useLiveCamera, type ConnectionState } from "@/hooks/use-live-camera";
import {
  detectImage,
  detectVideo,
  outputUrl,
  type VideoDetectResult,
} from "@/lib/api";
import type { FaceIdentity, ImageDetectResponse, RiskLevel, SafetyPayload } from "@/lib/types";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/monitor")({
  head: () => ({
    meta: [{ title: "Live Monitor & Detection | SentinelOps" }],
  }),
  component: MonitorPage,
});

type MonitorMode = "camera" | "video" | "image";

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

// Audio chime using Web Audio API (no external asset required)
function playAlertBeep() {
  try {
    const AudioContext = window.AudioContext || (window as unknown as { webkitAudioContext: typeof window.AudioContext }).webkitAudioContext;
    if (!AudioContext) return;
    const ctx = new AudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(880, ctx.currentTime);
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.3);
  } catch {
    /* ignore audio autoplay restrictions */
  }
}

function MonitorPage() {
  const [mode, setMode] = useState<MonitorMode>("camera");
  const [audioEnabled, setAudioEnabled] = useState(false);

  // Live Camera Hook
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

  // Audio alert trigger on HIGH / CRITICAL
  useEffect(() => {
    if (audioEnabled && connection === "connected" && (riskLevel === "HIGH" || riskLevel === "CRITICAL")) {
      playAlertBeep();
    }
  }, [audioEnabled, connection, riskLevel]);

  // Snapshot functionality
  const handleTakeSnapshot = () => {
    const imgEl = attachCanvasImage as unknown as { current: HTMLImageElement | null };
    if (!imgEl?.current) return;
    const link = document.createElement("a");
    link.href = imgEl.current.src;
    link.download = `safety_snapshot_${Date.now()}.jpg`;
    link.click();
  };

  // Video Detection State
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [isProcessingVideo, setIsProcessingVideo] = useState(false);
  const [videoResult, setVideoResult] = useState<VideoDetectResult | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);

  // Image Detection State
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [isProcessingImage, setIsProcessingImage] = useState(false);
  const [imageResult, setImageResult] = useState<ImageDetectResponse | null>(null);
  const [imageError, setImageError] = useState<string | null>(null);

  const handleVideoUpload = async (file: File) => {
    setVideoFile(file);
    setIsProcessingVideo(true);
    setVideoError(null);
    setVideoResult(null);
    try {
      const res = await detectVideo(file);
      setVideoResult(res);
    } catch (err: unknown) {
      setVideoError(err instanceof Error ? err.message : "Video detection failed");
    } finally {
      setIsProcessingVideo(false);
    }
  };

  const handleImageUpload = async (file: File) => {
    setImageFile(file);
    setIsProcessingImage(true);
    setImageError(null);
    setImageResult(null);
    try {
      const res = await detectImage(file);
      setImageResult(res);
    } catch (err: unknown) {
      setImageError(err instanceof Error ? err.message : "Image detection failed");
    } finally {
      setIsProcessingImage(false);
    }
  };

  return (
    <div>
      <PageHeader
        eyebrow="Safety monitoring & analysis"
        title="Detection center"
        description="Stream live webcam feeds, process recorded video clips, or analyze static images through the complete AI safety pipeline."
        actions={
          <div className="flex items-center gap-2">
            {mode === "camera" && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setAudioEnabled((prev) => !prev)}
                  title={audioEnabled ? "Mute alert audio chime" : "Enable alert audio chime on HIGH/CRITICAL"}
                >
                  {audioEnabled ? <Volume2 className="size-4 text-primary" /> : <VolumeX className="size-4" />}
                  {audioEnabled ? "Audio alert on" : "Mute alerts"}
                </Button>
                {connection === "connected" ? (
                  <>
                    <Button variant="outline" size="sm" onClick={handleTakeSnapshot} disabled={!hasFrame}>
                      <Camera className="size-4" /> Snapshot
                    </Button>
                    <Button variant="destructive" size="sm" onClick={stop}>
                      <CircleStop className="size-4" /> Stop camera
                    </Button>
                  </>
                ) : (
                  <Button size="sm" onClick={handleStart} disabled={connection === "connecting"}>
                    <Cctv className="size-4" />
                    {connection === "connecting" ? "Starting…" : "Start live camera"}
                  </Button>
                )}
              </>
            )}
          </div>
        }
      />

      {/* Mode navigation tabs */}
      <div className="mb-4 flex flex-wrap items-center gap-2 border-b pb-3">
        <button
          type="button"
          onClick={() => setMode("camera")}
          className={cn(
            "flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-semibold transition-colors",
            mode === "camera"
              ? "bg-primary text-primary-foreground shadow-sm"
              : "bg-card text-muted-foreground hover:bg-accent hover:text-foreground border",
          )}
        >
          <Cctv className="size-4" /> Live Webcam Stream
        </button>
        <button
          type="button"
          onClick={() => setMode("video")}
          className={cn(
            "flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-semibold transition-colors",
            mode === "video"
              ? "bg-primary text-primary-foreground shadow-sm"
              : "bg-card text-muted-foreground hover:bg-accent hover:text-foreground border",
          )}
        >
          <FileVideo className="size-4" /> Video File Analysis
        </button>
        <button
          type="button"
          onClick={() => setMode("image")}
          className={cn(
            "flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-semibold transition-colors",
            mode === "image"
              ? "bg-primary text-primary-foreground shadow-sm"
              : "bg-card text-muted-foreground hover:bg-accent hover:text-foreground border",
          )}
        >
          <FileImage className="size-4" /> Single Image Analysis
        </button>
      </div>

      {/* Mode 1: Live Webcam Stream */}
      {mode === "camera" && (
        <div>
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
                  <Radar className="size-4 text-primary" /> Live AI feed
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
                    {identitySummary.faces_seen} face{identitySummary.faces_seen === 1 ? "" : "s"} this frame · {identitySummary.workers_verified} verified · {identitySummary.unknown_faces} unknown
                  </p>
                ) : null}
              </SectionCard>

              <SectionCard
                title="PPE compliance"
                description="Hardhat / Vest / Mask compliance"
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

              <SectionCard title="Machinery & Proximity" description="Heavy machinery, distance, danger zones">
                <div className="space-y-2 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-1.5 text-muted-foreground">
                      <Truck className="size-3.5 text-primary" /> Machinery detected:
                    </span>
                    <span className="font-semibold text-foreground">{summary?.machine_count ?? 0}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-1.5 text-muted-foreground">
                      <AlertTriangle className="size-3.5 text-medium" /> Danger zones active:
                    </span>
                    <span className="font-semibold text-foreground">{dangerZones.length}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-1.5 text-muted-foreground">
                      <ShieldAlert className="size-3.5 text-high" /> Proximity alerts:
                    </span>
                    <span className="font-semibold text-foreground">{nearViolations}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="flex items-center gap-1.5 text-muted-foreground">
                      <Flame className="size-3.5 text-critical" /> Fire / Smoke:
                    </span>
                    <span className={cn("font-semibold", fireCount || smokeCount ? "text-critical" : "text-safe")}>
                      {fireCount} / {smokeCount}
                    </span>
                  </div>
                </div>
              </SectionCard>
            </div>
          </div>
        </div>
      )}

      {/* Mode 2: Video File Detection */}
      {mode === "video" && (
        <div className="space-y-4">
          <SectionCard
            title="Video detection pipeline"
            description="Upload recorded site footage to analyze every frame for hazards, machinery, PPE violations, and tracked individuals."
          >
            {!videoResult && !isProcessingVideo ? (
              <Dropzone
                accept="video/mp4,video/quicktime,video/x-msvideo,video/webm"
                label="Drag and drop site video file, or click to browse"
                sublabel="MP4, MOV, AVI, or WEBM up to 100MB"
                icon={FileVideo}
                onFileSelect={handleVideoUpload}
              />
            ) : null}

            {isProcessingVideo ? (
              <div className="py-12 text-center">
                <Loader2 className="mx-auto size-8 animate-spin text-primary" />
                <p className="mt-4 font-display text-sm font-semibold">Processing video through AI pipeline…</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Running YOLO object detector, ByteTrack tracking, fire/smoke model, and face recognition on every frame.
                </p>
              </div>
            ) : null}

            {videoError ? (
              <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-xs text-destructive">
                <p className="font-semibold">Video processing error:</p>
                <p className="mt-1">{videoError}</p>
                <Button size="sm" variant="outline" className="mt-3" onClick={() => setVideoError(null)}>
                  <RotateCcw className="size-3.5" /> Try again
                </Button>
              </div>
            ) : null}

            {videoResult ? (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-muted p-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold">Analysis complete:</span>
                    <RiskBadge level={videoResult.maxRiskLevel as RiskLevel} />
                  </div>
                  <div className="flex items-center gap-3 text-xs text-muted-foreground">
                    <span>Processed: <strong>{videoResult.framesProcessed}</strong> frames</span>
                    {videoResult.framesFailed > 0 && <span>Failed: <strong className="text-destructive">{videoResult.framesFailed}</strong></span>}
                    <a
                      href={videoResult.videoUrl}
                      download="safety_analysis.mp4"
                      className="inline-flex items-center gap-1 rounded bg-primary px-2.5 py-1 text-xs font-semibold text-primary-foreground hover:bg-primary/90"
                    >
                      <Download className="size-3.5" /> Download video
                    </a>
                    <Button size="sm" variant="outline" onClick={() => { setVideoResult(null); setVideoFile(null); }}>
                      <RotateCcw className="size-3.5" /> Upload another
                    </Button>
                  </div>
                </div>

                <div className="relative aspect-video w-full overflow-hidden rounded-lg border bg-black shadow">
                  <video src={videoResult.videoUrl} controls autoPlay className="h-full w-full object-contain" />
                </div>
              </div>
            ) : null}
          </SectionCard>
        </div>
      )}

      {/* Mode 3: Image Upload Detection */}
      {mode === "image" && (
        <div className="space-y-4">
          <SectionCard
            title="Image detection pipeline"
            description="Upload a high-resolution snapshot to detect hazards, machinery, PPE violations, danger zones, and verified workers."
          >
            {!imageResult && !isProcessingImage ? (
              <Dropzone
                accept="image/jpeg,image/png,image/webp"
                label="Drag and drop safety image, or click to browse"
                sublabel="JPEG, PNG, or WEBM"
                icon={FileImage}
                onFileSelect={handleImageUpload}
              />
            ) : null}

            {isProcessingImage ? (
              <div className="py-12 text-center">
                <Loader2 className="mx-auto size-8 animate-spin text-primary" />
                <p className="mt-4 font-display text-sm font-semibold">Analyzing image through safety pipeline…</p>
                <p className="mt-1 text-xs text-muted-foreground">Running hazard models, zone evaluation, and face identification.</p>
              </div>
            ) : null}

            {imageError ? (
              <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-4 text-xs text-destructive">
                <p className="font-semibold">Image processing error:</p>
                <p className="mt-1">{imageError}</p>
                <Button size="sm" variant="outline" className="mt-3" onClick={() => setImageError(null)}>
                  <RotateCcw className="size-3.5" /> Try again
                </Button>
              </div>
            ) : null}

            {imageResult && imageResult.result ? (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-muted p-3">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold">{imageResult.filename}:</span>
                    <RiskBadge level={imageResult.result.risk_level} />
                  </div>
                  <div className="flex items-center gap-2">
                    {imageResult.annotated_image_path ? (
                      <a
                        href={outputUrl(imageResult.annotated_image_path) ?? "#"}
                        target="_blank"
                        rel="noreferrer"
                        download
                        className="inline-flex items-center gap-1 rounded bg-primary px-2.5 py-1 text-xs font-semibold text-primary-foreground hover:bg-primary/90"
                      >
                        <Download className="size-3.5" /> Save image
                      </a>
                    ) : null}
                    <Button size="sm" variant="outline" onClick={() => { setImageResult(null); setImageFile(null); }}>
                      <RotateCcw className="size-3.5" /> Upload another
                    </Button>
                  </div>
                </div>

                <div className="grid gap-4 xl:grid-cols-2">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="font-display text-xs font-semibold">Annotated Output Image</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="overflow-hidden rounded-lg border bg-black">
                        <img
                          src={outputUrl(imageResult.annotated_image_path) ?? ""}
                          alt="Annotated detection output"
                          className="w-full object-contain max-h-[480px]"
                        />
                      </div>
                    </CardContent>
                  </Card>

                  <div className="space-y-3">
                    {/* Detections summary */}
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="font-display text-xs font-semibold">Detected Objects ({imageResult.result.detections.length})</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="flex flex-wrap gap-1.5">
                          {imageResult.result.detections.map((det, idx) => (
                            <span key={`${det.class}-${idx}`} className="rounded border bg-muted px-2 py-0.5 text-xs font-medium">
                              {det.class} <span className="text-muted-foreground">{(det.confidence * 100).toFixed(0)}%</span>
                            </span>
                          ))}
                        </div>
                      </CardContent>
                    </Card>

                    {/* Violations */}
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="font-display text-xs font-semibold text-high">
                          Violations ({imageResult.result.violation_count})
                        </CardTitle>
                      </CardHeader>
                      <CardContent>
                        {imageResult.result.violations.length > 0 ? (
                          <ul className="space-y-1.5 text-xs">
                            {imageResult.result.violations.map((v, idx) => (
                              <li key={idx} className="flex items-center gap-2 rounded bg-critical/5 px-2 py-1 text-critical font-medium">
                                <AlertTriangle className="size-3.5 shrink-0" />
                                {v.message ?? v.type}
                              </li>
                            ))}
                          </ul>
                        ) : (
                          <p className="text-xs text-safe">No violations found in this image.</p>
                        )}
                      </CardContent>
                    </Card>

                    {/* Identities */}
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="font-display text-xs font-semibold">Worker Identities</CardTitle>
                      </CardHeader>
                      <CardContent>
                        {imageResult.result.identities.length > 0 ? (
                          <ul className="space-y-2">
                            {imageResult.result.identities.map((id, idx) => (
                              <WorkerRow key={idx} identity={id} />
                            ))}
                          </ul>
                        ) : (
                          <p className="text-xs text-muted-foreground">No faces detected or verified.</p>
                        )}
                      </CardContent>
                    </Card>
                  </div>
                </div>
              </div>
            ) : null}
          </SectionCard>
        </div>
      )}
    </div>
  );
}

// Drag and drop file component
function Dropzone({
  accept,
  label,
  sublabel,
  icon: Icon,
  onFileSelect,
}: {
  accept: string;
  label: string;
  sublabel: string;
  icon: typeof FileVideo;
  onFileSelect: (file: File) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) onFileSelect(files[0]!);
  };

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (files.length > 0) onFileSelect(files[0]!);
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current?.click()}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 text-center transition-colors",
        isDragOver ? "border-primary bg-primary/5" : "border-border hover:border-primary/50 hover:bg-accent/50",
      )}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        hidden
        onChange={handleChange}
        aria-label={label}
      />
      <span className="grid size-12 place-items-center rounded-full bg-primary/10 text-primary">
        <Icon className="size-6" />
      </span>
      <p className="mt-3 font-display text-sm font-semibold">{label}</p>
      <p className="mt-1 text-xs text-muted-foreground">{sublabel}</p>
      <Button size="sm" variant="outline" className="mt-4 pointer-events-none">
        <UploadCloud className="size-4" /> Select File
      </Button>
    </div>
  );
}
