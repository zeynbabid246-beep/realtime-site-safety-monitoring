import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

/* ------------------------------------------------------------------ */
/* Stat card                                                           */
/* ------------------------------------------------------------------ */

export function StatCard({
  label,
  value,
  detail,
  icon: Icon,
  tone = "neutral",
  className,
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  icon?: LucideIcon;
  tone?: "neutral" | "success" | "warning" | "danger" | "info";
  className?: string;
}) {
  const iconTones = {
    neutral: "bg-muted text-muted-foreground",
    success: "bg-safe/12 text-safe",
    warning: "bg-medium/15 text-medium",
    danger: "bg-critical/12 text-critical",
    info: "bg-primary/10 text-primary",
  } as const;

  return (
    <Card className={cn("panel-rise", className)}>
      <CardContent className="flex items-start justify-between gap-3 pt-0">
        <div className="min-w-0">
          <p className="truncate text-[11px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
            {label}
          </p>
          <p className="mt-2 font-display text-3xl font-bold leading-none">{value}</p>
          {detail ? (
            <p className="mt-1.5 truncate text-xs text-muted-foreground">{detail}</p>
          ) : null}
        </div>
        {Icon ? (
          <span
            className={cn("grid size-9 shrink-0 place-items-center rounded-lg", iconTones[tone])}
          >
            <Icon className="size-4.5" />
          </span>
        ) : null}
      </CardContent>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Section card                                                        */
/* ------------------------------------------------------------------ */

export function SectionCard({
  title,
  description,
  action,
  children,
  className,
  contentClassName,
}: {
  title: string;
  description?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
  contentClassName?: string;
}) {
  return (
    <Card className={cn("panel-rise", className)}>
      <CardHeader className="flex-row items-start justify-between space-y-0">
        <div className="min-w-0">
          <CardTitle className="font-display text-sm font-semibold">{title}</CardTitle>
          {description ? <p className="mt-1 text-xs text-muted-foreground">{description}</p> : null}
        </div>
        {action}
      </CardHeader>
      <CardContent className={contentClassName}>{children}</CardContent>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* States                                                              */
/* ------------------------------------------------------------------ */

export function EmptyState({
  icon: Icon,
  title,
  description,
  action,
  className,
}: {
  icon: LucideIcon;
  title: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center rounded-lg border border-dashed px-6 py-10 text-center",
        className,
      )}
    >
      <span className="grid size-11 place-items-center rounded-full bg-muted">
        <Icon className="size-5 text-muted-foreground" />
      </span>
      <p className="mt-3 text-sm font-semibold">{title}</p>
      {description ? (
        <p className="mt-1 max-w-sm text-xs leading-relaxed text-muted-foreground">{description}</p>
      ) : null}
      {action ? <div className="mt-4">{action}</div> : null}
    </div>
  );
}

export function ErrorState({
  message,
  onRetry,
  className,
}: {
  message: string;
  onRetry?: () => void;
  className?: string;
}) {
  return (
    <div
      role="alert"
      className={cn(
        "flex flex-col items-center justify-center rounded-lg border border-destructive/30 bg-destructive/5 px-6 py-8 text-center",
        className,
      )}
    >
      <span className="grid size-11 place-items-center rounded-full bg-destructive/10">
        <AlertTriangle className="size-5 text-destructive" />
      </span>
      <p className="mt-3 text-sm font-semibold">Something went wrong</p>
      <p className="mt-1 max-w-md text-xs leading-relaxed text-muted-foreground">{message}</p>
      {onRetry ? (
        <Button variant="outline" size="sm" className="mt-4" onClick={onRetry}>
          <RefreshCw className="size-3.5" /> Retry
        </Button>
      ) : null}
    </div>
  );
}

export function LoadingCard({ rows = 3, className }: { rows?: number; className?: string }) {
  return (
    <Card className={className}>
      <CardContent className="space-y-3 pt-0">
        <Skeleton className="skeleton-shimmer h-4 w-1/3" />
        {Array.from({ length: rows }, (_, index) => (
          <Skeleton key={index} className="skeleton-shimmer h-10 w-full" />
        ))}
      </CardContent>
    </Card>
  );
}

export function LoadingStatCards({ count = 4 }: { count?: number }) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
      {Array.from({ length: count }, (_, index) => (
        <Card key={index}>
          <CardContent className="pt-0">
            <Skeleton className="skeleton-shimmer h-3 w-20" />
            <Skeleton className="skeleton-shimmer mt-3 h-8 w-16" />
            <Skeleton className="skeleton-shimmer mt-3 h-3 w-24" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
