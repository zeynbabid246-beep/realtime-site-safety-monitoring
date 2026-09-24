/**
 * Typed mirrors of the backend contracts (app/main.py, src/pipeline.py,
 * src/face_recognition/service.py, app/storage.py).
 *
 * These are hand-maintained but deliberately conservative: optional fields
 * default to safe fallbacks in the UI so a partial payload never breaks
 * rendering.
 */

/* ------------------------------------------------------------------ */
/* Health                                                              */
/* ------------------------------------------------------------------ */

export interface HealthResponse {
  status: string;
  version: string;
}

/* ------------------------------------------------------------------ */
/* Safety pipeline payloads (websocket + REST image endpoint)          */
/* ------------------------------------------------------------------ */

export type RiskLevel = "SAFE" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export const RISK_LEVELS: readonly RiskLevel[] = ["SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL"];

export interface Violation {
  type: string;
  severity?: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  track_id?: number | null;
  confidence?: number;
  message?: string;
  [key: string]: unknown;
}

/** One verified/unverified face, from FaceIdentity.to_dict(). */
export interface FaceIdentity {
  track_id: number | null;
  person_index: number;
  bbox: number[];
  verified: boolean;
  worker_id: string | null;
  worker_name: string | null;
  label: string;
  score: number; // winning worker's rate 0..1
  mean_score: number; // mean pair score 0..1
  margin: number | null;
  matches: Array<{
    worker_id: string;
    worker_name: string;
    rate: number;
    mean_score: number;
    detections: number;
    n_references: number;
    verified: boolean;
  }>;
  error: string | null;
}

export interface IdentitySummary {
  faces_seen: number;
  workers_verified: number;
  unknown_faces: number;
  verified_workers: string[];
}

export interface DangerZoneInfo {
  zone_id: number | null;
  age?: number;
  missed?: number;
  area: number | null;
  coordinates: number[][];
}

export interface DistanceResult {
  track_id?: number | null;
  machine_class?: string;
  distance_px?: number | null;
  distance_m_est?: number | null;
  [key: string]: unknown;
}

export interface FireDetection {
  type?: string;
  class_name?: string;
  confidence: number;
  bbox?: number[] | null;
}

export interface DetectionSummary {
  risk_level: RiskLevel;
  violation_count: number;
  violation_counts: Record<string, number>;
  ppe_violations: number;
  fire_count: number;
  smoke_count: number;
  person_count: number;
  machine_count: number;
  danger_zones: number;
}

export interface UiDetection {
  class: string;
  confidence: number;
}

/** The full websocket / image-endpoint JSON contract. */
export interface SafetyPayload {
  risk_level: RiskLevel;
  violations: Violation[];
  violation_count: number;
  violation_counts: Record<string, number>;
  danger_zones: DangerZoneInfo[];
  people_inside_zones: Array<Record<string, unknown>>;
  distance_results: DistanceResult[];
  pole_results: Array<Record<string, unknown>>;
  fire_detections: FireDetection[];
  statistics: Record<string, number>;
  summary: DetectionSummary;
  detections: UiDetection[];
  identities: FaceIdentity[];
  identity_summary: IdentitySummary;
}

/* ------------------------------------------------------------------ */
/* Events / alerts (SQLite persistence)                                */
/* ------------------------------------------------------------------ */

export interface SafetyEvent {
  id: number;
  ts: number; // epoch seconds
  source: string;
  risk_level: RiskLevel;
  violation_count: number;
  persons: number;
  machines: number;
  zones: number;
  fire: number;
  smoke: number;
  ppe: number;
  violation_counts: Record<string, number>;
  summary?: DetectionSummary;
  evidence_image: string | null;
  evidence_clip: string | null;
}

export interface SafetyAlert {
  id: number;
  ts: number;
  event_id: number | null;
  level: RiskLevel;
  title: string;
  message: string;
  channel: string;
  status: "new" | "acknowledged";
  notified: 0 | 1;
}

export interface EventsResponse {
  events: SafetyEvent[];
  count: number;
}

export interface AlertsResponse {
  alerts: SafetyAlert[];
  count: number;
  unacknowledged: number;
}

/* ------------------------------------------------------------------ */
/* Reports / statistics                                                */
/* ------------------------------------------------------------------ */

export type ReportRange = "24h" | "7d" | "30d" | "all";

export interface ReportTotals {
  events: number;
  alerts: number;
  alerts_acknowledged: number;
  alerts_unacknowledged: number;
  frames_processed: number;
  events_with_image: number;
  events_with_clip: number;
}

export interface SiteReport {
  range: ReportRange;
  generated_at: number;
  since: number | null;
  totals: ReportTotals;
  by_risk: Record<RiskLevel, number>;
  by_violation_type: Record<string, number>;
  by_source: Record<string, number>;
  by_day: Record<string, number>;
  top_events: Array<
    Pick<
      SafetyEvent,
      "id" | "ts" | "source" | "risk_level" | "violation_count" | "violation_counts"
    > & {
      evidence_image: string | null;
      evidence_clip: string | null;
    }
  >;
}

/* ------------------------------------------------------------------ */
/* Face recognition / workers (app/main.py /face/* endpoints)          */
/* ------------------------------------------------------------------ */

export interface WorkerRecord {
  worker_id: string;
  name: string;
  role: string;
  active: boolean;
  created_at: number;
  updated_at: number;
  meta: Record<string, unknown>;
  reference_images: string[];
}

export interface WorkersResponse {
  success: boolean;
  workers: WorkerRecord[];
  count: number;
}

export interface RegisterWorkerResponse {
  success: boolean;
  worker?: WorkerRecord;
  images_added?: number;
  n_references?: number;
  error?: string;
}

export interface RemoveWorkerResponse {
  success: boolean;
  worker_id: string;
  purged: boolean;
}

export interface FaceVerifyResponse {
  success: boolean;
  filename?: string;
  faces?: FaceIdentity[];
  verified_workers?: Array<{ worker_id: string; worker_name: string; score: number }>;
  error?: string;
}

export interface CalibrationResult {
  suggested_detection_threshold: number;
  suggested_verification_threshold: number;
  true_accept_rate_at_suggestion: number;
  false_accept_rate_at_suggestion: number;
  genuine_min?: number | null;
  genuine_mean?: number | null;
  impostor_max?: number | null;
  n_genuine_pairs: number;
  n_impostor_pairs: number;
  calibrated_at: number;
}

export interface CalibrateResponse {
  success: boolean;
  calibration?: CalibrationResult;
  error?: string;
}

/* ------------------------------------------------------------------ */
/* Image detection endpoint                                            */
/* ------------------------------------------------------------------ */

export interface ImageDetectResponse {
  success: boolean;
  filename?: string;
  result?: SafetyPayload;
  annotated_image_path?: string;
  error?: string;
}
