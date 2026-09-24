import type { RiskLevel } from "@/lib/types";
import { cn } from "@/lib/utils";

/** Tailwind classes per risk level — shared by every badge in the app. */
export const RISK_STYLES: Record<RiskLevel, string> = {
  SAFE: "bg-safe/12 text-safe ring-safe/30",
  LOW: "bg-low/12 text-low ring-low/30",
  MEDIUM: "bg-medium/15 text-medium ring-medium/30",
  HIGH: "bg-high/15 text-high ring-high/35",
  CRITICAL: "bg-critical/15 text-critical ring-critical/35",
};

export const RISK_LABELS: Record<RiskLevel, string> = {
  SAFE: "Safe",
  LOW: "Low",
  MEDIUM: "Medium",
  HIGH: "High",
  CRITICAL: "Critical",
};

const RISK_DOTS: Record<RiskLevel, string> = {
  SAFE: "bg-safe",
  LOW: "bg-low",
  MEDIUM: "bg-medium",
  HIGH: "bg-high",
  CRITICAL: "bg-critical",
};

export function RiskBadge({
  level,
  size = "md",
  pulse = false,
  className,
}: {
  level: RiskLevel;
  size?: "sm" | "md";
  pulse?: boolean;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full font-semibold uppercase tracking-wide ring-1 ring-inset",
        size === "sm" ? "px-2 py-0.5 text-[10px]" : "px-2.5 py-1 text-[11px]",
        RISK_STYLES[level],
        className,
      )}
    >
      <span className={cn("size-1.5 rounded-full", RISK_DOTS[level], pulse && "status-pulse")} />
      {RISK_LABELS[level]}
    </span>
  );
}

export function StatusBadge({
  tone = "neutral",
  children,
  className,
}: {
  tone?: "neutral" | "success" | "warning" | "danger" | "info";
  children: React.ReactNode;
  className?: string | undefined;
}) {
  const tones = {
    neutral: "bg-muted text-muted-foreground ring-border",
    success: "bg-safe/12 text-safe ring-safe/30",
    warning: "bg-medium/15 text-medium ring-medium/30",
    danger: "bg-critical/15 text-critical ring-critical/35",
    info: "bg-primary/10 text-primary ring-primary/30",
  } as const;

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10.5px] font-semibold uppercase tracking-wide ring-1 ring-inset",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/** Verified / Unknown identity badge for worker face recognition. */
export function IdentityBadge({
  verified,
  name,
  className,
}: {
  verified: boolean;
  name?: string | null;
  className?: string | undefined;
}) {
  const label = verified ? (name ? `Verified · ${name}` : "Verified") : "Unknown";
  return (
    <StatusBadge tone={verified ? "success" : "neutral"} className={className}>
      {label}
    </StatusBadge>
  );
}
