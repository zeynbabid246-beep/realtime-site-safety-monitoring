import type { ReactNode } from "react";
import { SiteHeader } from "@/components/site-header";

/**
 * App shell: dark navy sidebar + sticky header + content area.
 * Every route renders inside this shell.
 */
export function AppShell({ children }: { children: ReactNode }) {
  return <SiteHeader>{children}</SiteHeader>;
}
