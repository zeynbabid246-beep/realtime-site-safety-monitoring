/**
 * Typed API client for the construction-safety backend (app/main.py).
 *
 * Base URL resolution mirrors the previous vanilla dashboard: same-origin
 * when the app is served by FastAPI, otherwise http://<host>:8000 in
 * development. Override with VITE_API_BASE if needed.
 */

import type {
  AlertsResponse,
  CalibrateResponse,
  EventsResponse,
  FaceVerifyResponse,
  HealthResponse,
  ImageDetectResponse,
  RegisterWorkerResponse,
  RemoveWorkerResponse,
  ReportRange,
  SafetyEvent,
  SiteReport,
  WorkersResponse,
} from "./types";

export function getApiBase(): string {
  const override = import.meta.env["VITE_API_BASE"] as string | undefined;
  if (override) return override.replace(/\/$/, "");

  if (typeof window !== "undefined") {
    const { protocol, hostname } = window.location;
    // Same-origin when served by FastAPI (production) or an SSR server.
    const isDevServer = hostname === "localhost" || hostname === "127.0.0.1";
    if (!isDevServer) return `${protocol}//${window.location.host}`;
    // Vite dev server (5173) or another frontend-only host -> backend on :8000.
    return `${protocol}//${hostname}:8000`;
  }
  return "http://127.0.0.1:8000";
}

export function evidenceUrl(relPath: string | null | undefined): string | null {
  if (!relPath) return null;
  const filename = relPath.split(/[\\/]/).pop() ?? relPath;
  return `${getApiBase()}/evidence/${encodeURIComponent(filename)}`;
}

export function outputUrl(path: string | null | undefined): string | null {
  if (!path) return null;
  const filename = path.split(/[\\/]/).pop() ?? path;
  return `${getApiBase()}/output/${encodeURIComponent(filename)}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBase()}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { error?: string; detail?: unknown };
      if (body.error) detail = body.error;
      else if (typeof body.detail === "string") detail = body.detail;
      else if (body.detail) detail = JSON.stringify(body.detail);
    } catch {
      /* keep status-text detail */
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

/* ------------------------------------------------------------------ */
/* Health                                                              */
/* ------------------------------------------------------------------ */

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

/* ------------------------------------------------------------------ */
/* Events / alerts                                                     */
/* ------------------------------------------------------------------ */

export interface EventFilters {
  limit?: number;
  risk?: string;
  source?: string;
  since?: number;
}

export function listEvents(filters: EventFilters = {}): Promise<EventsResponse> {
  const params = new URLSearchParams();
  if (filters.limit != null) params.set("limit", String(filters.limit));
  if (filters.risk) params.set("risk", filters.risk);
  if (filters.source) params.set("source", filters.source);
  if (filters.since != null) params.set("since", String(filters.since));
  const qs = params.toString();
  return request<EventsResponse>(`/api/events${qs ? `?${qs}` : ""}`);
}

export interface AlertFilters {
  limit?: number;
  status?: "new" | "acknowledged";
  level?: string;
  since?: number;
}

export function listAlerts(filters: AlertFilters = {}): Promise<AlertsResponse> {
  const params = new URLSearchParams();
  if (filters.limit != null) params.set("limit", String(filters.limit));
  if (filters.status) params.set("status", filters.status);
  if (filters.level) params.set("level", filters.level);
  if (filters.since != null) params.set("since", String(filters.since));
  const qs = params.toString();
  return request<AlertsResponse>(`/api/alerts${qs ? `?${qs}` : ""}`);
}

export function acknowledgeAlert(alertId: number): Promise<{ success: boolean; id: number }> {
  return request(`/api/alerts/${alertId}/ack`, { method: "POST" });
}

export function acknowledgeAllAlerts(): Promise<{ success: boolean; acknowledged_count: number }> {
  return request("/api/alerts/ack-all", { method: "POST" });
}

/* ------------------------------------------------------------------ */
/* Statistics / reports                                                */
/* ------------------------------------------------------------------ */

export function getReport(range: ReportRange): Promise<SiteReport> {
  return request<SiteReport>(`/api/statistics?range=${encodeURIComponent(range)}`);
}

export function downloadReport(range: ReportRange, format: "json" | "csv"): Promise<Response> {
  return fetch(`${getApiBase()}/api/reports/download?range=${range}&format=${format}`);
}

/* ------------------------------------------------------------------ */
/* Evidence gallery                                                    */
/* ------------------------------------------------------------------ */

export function listEvidence(limit = 60): Promise<EventsResponse> {
  return request<EventsResponse>(`/api/evidence?limit=${limit}`);
}

/* ------------------------------------------------------------------ */
/* Face recognition / workers                                          */
/* ------------------------------------------------------------------ */

export function listWorkers(): Promise<WorkersResponse> {
  return request<WorkersResponse>("/face/workers");
}

export interface RegisterWorkerInput {
  workerId: string;
  name?: string;
  role?: string;
  files: File[];
}

export async function registerWorker(input: RegisterWorkerInput): Promise<RegisterWorkerResponse> {
  const params = new URLSearchParams({ worker_id: input.workerId });
  if (input.name) params.set("name", input.name);
  if (input.role) params.set("role", input.role);

  const form = new FormData();
  for (const file of input.files) form.append("files", file);

  return request<RegisterWorkerResponse>(`/face/workers?${params.toString()}`, {
    method: "POST",
    body: form,
  });
}

export async function addWorkerImages(
  workerId: string,
  files: File[],
): Promise<RegisterWorkerResponse> {
  const form = new FormData();
  for (const file of files) form.append("files", file);
  return request<RegisterWorkerResponse>(`/face/workers/${encodeURIComponent(workerId)}/images`, {
    method: "POST",
    body: form,
  });
}

export interface RemoveWorkerOptions {
  deleteImages?: boolean;
}

export function removeWorker(
  workerId: string,
  options: RemoveWorkerOptions = {},
): Promise<RemoveWorkerResponse> {
  const params = new URLSearchParams();
  if (options.deleteImages) params.set("delete_images", "true");
  const qs = params.toString();
  return request<RemoveWorkerResponse>(
    `/face/workers/${encodeURIComponent(workerId)}${qs ? `?${qs}` : ""}`,
    { method: "DELETE" },
  );
}

export async function verifyFaceImage(file: File): Promise<FaceVerifyResponse> {
  const form = new FormData();
  form.append("file", file);
  return request<FaceVerifyResponse>("/face/verify", { method: "POST", body: form });
}

export function calibrateThresholds(): Promise<CalibrateResponse> {
  return request<CalibrateResponse>("/face/calibrate", { method: "POST" });
}

/* ------------------------------------------------------------------ */
/* Image & Video detection                                             */
/* ------------------------------------------------------------------ */

export async function detectImage(file: File): Promise<ImageDetectResponse> {
  const form = new FormData();
  form.append("file", file);
  return request<ImageDetectResponse>("/safety/detect/image", { method: "POST", body: form });
}

export interface VideoDetectResult {
  videoUrl: string;
  framesProcessed: number;
  framesFailed: number;
  maxRiskLevel: string;
}

export async function detectVideo(file: File): Promise<VideoDetectResult> {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch(`${getApiBase()}/safety/detect/video`, {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { error?: string };
      if (body.error) detail = body.error;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }

  const blob = await response.blob();
  const videoUrl = URL.createObjectURL(blob);
  const framesProcessed = parseInt(response.headers.get("X-Frames-Processed") || "0", 10);
  const framesFailed = parseInt(response.headers.get("X-Frames-Failed") || "0", 10);
  const maxRiskLevel = response.headers.get("X-Max-Risk-Level") || "SAFE";

  return { videoUrl, framesProcessed, framesFailed, maxRiskLevel };
}

/* ------------------------------------------------------------------ */
/* WebSocket live camera                                               */
/* ------------------------------------------------------------------ */

export function getWebSocketUrl(): string {
  if (typeof window === "undefined") return "ws://127.0.0.1:8000/safety/ws/camera";
  const override = import.meta.env["VITE_WS_BASE"] as string | undefined;
  if (override) return override;
  const { protocol, hostname } = window.location;
  const wsProtocol = protocol === "https:" ? "wss:" : "ws:";
  return `${wsProtocol}//${hostname}:8000/safety/ws/camera`;
}

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

export function formatEventTime(ts: number): string {
  return new Date(ts * 1000).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export function relativeTime(ts: number): string {
  const diffSeconds = Math.max(0, Date.now() / 1000 - ts);
  if (diffSeconds < 60) return `${Math.floor(diffSeconds)}s ago`;
  if (diffSeconds < 3600) return `${Math.floor(diffSeconds / 60)}m ago`;
  if (diffSeconds < 86400) return `${Math.floor(diffSeconds / 3600)}h ago`;
  return `${Math.floor(diffSeconds / 86400)}d ago`;
}
