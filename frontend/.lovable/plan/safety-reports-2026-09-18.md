# Safety Reports

## Goal
Add a complete Reports view inside SentinelOps that filters safety performance by date and exports the filtered result as PDF or CSV.

## What will be built
- Turn the existing Reports navigation into a real report workspace without changing the live-monitoring dashboard.
- Add preset and custom date filters with clear active-range feedback.
- Show filtered incident totals, severity and status summaries, response metrics, and incident-type breakdowns.
- Add PPE compliance trend charts for hard hats, hi-vis vests, and footwear.
- Add site-level comparison rows for safety score, incidents, PPE compliance, and trend versus the previous period.
- Add working CSV and PDF download actions using the same filtered dataset shown on screen.
- Keep the current Control Room palette, Sora/Manrope typography, compact command-center density, and mobile behavior.

## Technical details
- Use a typed local historical dataset so the reporting workflow is immediately usable without a backend dependency.
- Generate CSV in the browser with safe escaping and a date-specific filename.
- Generate a branded multi-section PDF with `jsPDF`, including summary, incident detail, PPE trends, and site comparison tables.
- Keep downloads client-side and verify file creation, report interactions, responsive layout, and build/runtime status.
