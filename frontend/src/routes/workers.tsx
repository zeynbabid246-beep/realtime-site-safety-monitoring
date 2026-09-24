import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState, type ChangeEvent, type FormEvent } from "react";
import {
  CalendarClock,
  FileImage,
  Images,
  Loader2,
  ScanFace,
  Search,
  Trash2,
  Upload,
  UserPlus,
  UserRoundCheck,
  UserRoundX,
} from "lucide-react";
import { PageHeader } from "@/components/page-header";
import { StatusBadge } from "@/components/status-badge";
import { EmptyState, ErrorState, LoadingCard } from "@/components/stat-card";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { addWorkerImages, listWorkers, registerWorker, removeWorker } from "@/lib/api";
import type { WorkerRecord } from "@/lib/types";

export const Route = createFileRoute("/workers")({
  head: () => ({
    meta: [
      { title: "Workers & Identity | SentinelOps" },
      {
        name: "description",
        content: "Manage registered site workers and reference face images for live Siamese model verification.",
      },
    ],
  }),
  component: WorkersPage,
});

function formatTimestamp(ts: number): string {
  return new Date(ts * 1000).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function WorkerCard({
  worker,
  onAddImages,
  onRemove,
  busy,
}: {
  worker: WorkerRecord;
  onAddImages: (workerId: string, files: File[]) => void;
  onRemove: (worker: WorkerRecord) => void;
  busy: boolean;
}) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    if (files.length > 0) onAddImages(worker.worker_id, files);
    event.target.value = "";
  };

  return (
    <article className="panel-rise flex flex-col rounded-lg border bg-card p-4 shadow-sm hover:border-primary/30 transition-all">
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <span className="grid size-10 shrink-0 place-items-center rounded-full bg-primary/10 text-primary font-bold">
            <ScanFace className="size-5" />
          </span>
          <div className="min-w-0">
            <h3 className="truncate font-display text-sm font-semibold text-foreground">
              {worker.name || worker.worker_id}
            </h3>
            <p className="truncate text-xs text-muted-foreground">
              {worker.role || "Site Personnel"} · <code className="text-[10.5px] bg-muted px-1.5 py-0.5 rounded font-mono">{worker.worker_id}</code>
            </p>
          </div>
        </div>
        <StatusBadge tone={worker.active ? "success" : "neutral"}>
          {worker.active ? "Active" : "Deactivated"}
        </StatusBadge>
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-3 text-xs">
        <div className="rounded-md bg-muted/60 px-3 py-2">
          <dt className="flex items-center gap-1.5 text-muted-foreground font-medium">
            <FileImage className="size-3.5" /> Reference Faces
          </dt>
          <dd className="mt-1 font-display text-lg font-bold text-foreground">{worker.reference_images.length}</dd>
        </div>
        <div className="rounded-md bg-muted/60 px-3 py-2">
          <dt className="flex items-center gap-1.5 text-muted-foreground font-medium">
            <CalendarClock className="size-3.5" /> Registered
          </dt>
          <dd className="mt-1 font-semibold text-foreground">{formatTimestamp(worker.created_at)}</dd>
        </div>
      </dl>

      {worker.reference_images.length > 0 ? (
        <p className="mt-3 flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <Images className="size-3.5 text-primary" />
          {worker.reference_images.length} frontal crop sample{worker.reference_images.length === 1 ? "" : "s"} enrolled
        </p>
      ) : (
        <p className="mt-3 rounded-md bg-medium/10 px-2.5 py-1.5 text-[11px] text-medium font-medium">
          No reference faces enrolled — add photos to enable live webcam identity matching.
        </p>
      )}

      <div className="mt-4 flex items-center gap-2 border-t pt-3">
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          multiple
          hidden
          onChange={handleFileChange}
          aria-label={`Add reference face images for ${worker.name || worker.worker_id}`}
        />
        <Button
          size="sm"
          variant="outline"
          disabled={busy}
          onClick={() => fileInputRef.current?.click()}
        >
          <Upload className="size-3.5" /> Enroll Faces
        </Button>
        <Button
          size="sm"
          variant="ghost"
          className="ml-auto text-destructive hover:bg-destructive/10 hover:text-destructive"
          disabled={busy}
          onClick={() => onRemove(worker)}
        >
          <Trash2 className="size-3.5" /> Remove
        </Button>
      </div>
    </article>
  );
}

function WorkersPage() {
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");
  const [showInactive, setShowInactive] = useState(true);
  const [registerOpen, setRegisterOpen] = useState(false);
  const [removeTarget, setRemoveTarget] = useState<WorkerRecord | null>(null);
  const [purge, setPurge] = useState(false);
  const [form, setForm] = useState({ workerId: "", name: "", role: "" });
  const [files, setFiles] = useState<File[]>([]);
  const [formError, setFormError] = useState<string | null>(null);

  const workersQuery = useQuery({ queryKey: ["workers"], queryFn: listWorkers, retry: 1 });

  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: ["workers"] });
  };

  const registerMutation = useMutation({
    mutationFn: registerWorker,
    onSuccess: () => {
      setRegisterOpen(false);
      setForm({ workerId: "", name: "", role: "" });
      setFiles([]);
      setFormError(null);
      invalidate();
    },
    onError: (error: Error) => setFormError(error.message),
  });

  const addImagesMutation = useMutation({
    mutationFn: ({ workerId, files }: { workerId: string; files: File[] }) =>
      addWorkerImages(workerId, files),
    onSuccess: invalidate,
  });

  const removeMutation = useMutation({
    mutationFn: ({ workerId, deleteImages }: { workerId: string; deleteImages: boolean }) =>
      removeWorker(workerId, { deleteImages }),
    onSuccess: () => {
      setRemoveTarget(null);
      setPurge(false);
      invalidate();
    },
  });

  const workers = workersQuery.data?.workers ?? [];
  const filtered = workers
    .filter((worker) => (showInactive ? true : worker.active))
    .filter((worker) => {
      const needle = search.trim().toLowerCase();
      if (!needle) return true;
      return (
        worker.worker_id.toLowerCase().includes(needle) ||
        worker.name.toLowerCase().includes(needle) ||
        worker.role.toLowerCase().includes(needle)
      );
    });

  const handleRegisterSubmit = (event: FormEvent) => {
    event.preventDefault();
    if (!form.workerId.trim()) {
      setFormError("Worker ID is required.");
      return;
    }
    setFormError(null);
    registerMutation.mutate({
      workerId: form.workerId.trim(),
      name: form.name.trim(),
      role: form.role.trim(),
      files,
    });
  };

  const busy =
    registerMutation.isPending || addImagesMutation.isPending || removeMutation.isPending;

  return (
    <div>
      <PageHeader
        eyebrow="Identity & Access Management"
        title="Worker Verification Portal"
        description="Enrolled worker face crops are matched live on camera using our Siamese neural network backend. Each profile benefits from 5–10 varied lighting face crops."
        actions={
          <Button onClick={() => setRegisterOpen(true)}>
            <UserPlus className="size-4" /> Register New Worker
          </Button>
        }
      />

      {/* Toolbar */}
      <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center">
        <div className="relative flex-1 sm:max-w-xs">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search by ID, name, or role…"
            className="pl-9 text-xs"
            aria-label="Search workers"
          />
        </div>
        <label className="flex cursor-pointer items-center gap-2 text-xs font-medium text-muted-foreground">
          <input
            type="checkbox"
            checked={showInactive}
            onChange={(event) => setShowInactive(event.target.checked)}
            className="size-4 rounded border-input accent-[var(--primary)]"
          />
          Show deactivated profiles
        </label>
        <span className="ml-auto text-xs text-muted-foreground font-medium">
          {workers.length} enrolled · {workers.filter((worker) => worker.active).length} active on site
        </span>
      </div>

      {/* Content states */}
      {workersQuery.isLoading ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          <LoadingCard rows={2} />
          <LoadingCard rows={2} />
          <LoadingCard rows={2} />
        </div>
      ) : workersQuery.isError ? (
        <ErrorState
          message="The face recognition backend service is unavailable. Verify TensorFlow and model files on the server."
          onRetry={() => void workersQuery.refetch()}
        />
      ) : filtered.length === 0 ? (
        <EmptyState
          icon={ScanFace}
          title={search ? "No personnel match your search query" : "No registered workers on file"}
          description={
            search
              ? "Try searching by a different name, role, or ID string."
              : "Register personnel profiles with 5–10 reference face crops to activate real-time identification."
          }
          action={
            search ? undefined : (
              <Button onClick={() => setRegisterOpen(true)}>
                <UserPlus className="size-4" /> Register Worker Profile
              </Button>
            )
          }
        />
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {filtered.map((worker) => (
            <WorkerCard
              key={worker.worker_id}
              worker={worker}
              busy={busy}
              onAddImages={(workerId, imageFiles) =>
                addImagesMutation.mutate({ workerId, files: imageFiles })
              }
              onRemove={setRemoveTarget}
            />
          ))}
        </div>
      )}

      {/* Register dialog */}
      <Dialog open={registerOpen} onOpenChange={setRegisterOpen}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>Register Worker Profile</DialogTitle>
            <DialogDescription>
              Create a personnel record and upload initial reference face crops for live identification on site feeds.
            </DialogDescription>
          </DialogHeader>
          <form onSubmit={handleRegisterSubmit} className="space-y-4">
            <div className="grid gap-2">
              <Label htmlFor="worker-id">Worker ID / Badge Number *</Label>
              <Input
                id="worker-id"
                value={form.workerId}
                onChange={(event) =>
                  setForm((current) => ({ ...current, workerId: event.target.value }))
                }
                placeholder="e.g. worker_001"
                autoComplete="off"
                required
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="worker-name">Full Name</Label>
              <Input
                id="worker-name"
                value={form.name}
                onChange={(event) =>
                  setForm((current) => ({ ...current, name: event.target.value }))
                }
                placeholder="e.g. Marcus Vance"
                autoComplete="off"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="worker-role">Role / Job Title</Label>
              <Input
                id="worker-role"
                value={form.role}
                onChange={(event) =>
                  setForm((current) => ({ ...current, role: event.target.value }))
                }
                placeholder="e.g. Heavy Equipment Operator"
                autoComplete="off"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="worker-faces">Reference Face Crops</Label>
              <Input
                id="worker-faces"
                type="file"
                accept="image/jpeg,image/png,image/webp"
                multiple
                onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
              />
              <p className="text-[11px] text-muted-foreground">
                {files.length > 0
                  ? `${files.length} photo${files.length === 1 ? "" : "s"} selected`
                  : "Upload frontal face crops under clear lighting for high matching confidence."}
              </p>
            </div>
            {formError ? (
              <p
                role="alert"
                className="rounded-md bg-destructive/10 px-3 py-2 text-xs text-destructive font-medium"
              >
                {formError}
              </p>
            ) : null}
            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setRegisterOpen(false)}>
                Cancel
              </Button>
              <Button type="submit" disabled={registerMutation.isPending}>
                {registerMutation.isPending ? (
                  <>
                    <Loader2 className="size-4 animate-spin" /> Saving…
                  </>
                ) : (
                  <>
                    <UserPlus className="size-4" /> Save Worker Profile
                  </>
                )}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* Remove confirmation */}
      <Dialog
        open={removeTarget !== null}
        onOpenChange={(open) => {
          if (!open) setRemoveTarget(null);
        }}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Remove or Deactivate Worker</DialogTitle>
            <DialogDescription>
              Deactivating retains historical detection records while disabling future identity verification.
            </DialogDescription>
          </DialogHeader>
          {removeTarget ? (
            <div className="space-y-3">
              <div className="rounded-md bg-muted/60 px-3 py-2 text-sm">
                <p className="font-semibold text-foreground">{removeTarget.name || removeTarget.worker_id}</p>
                <p className="text-xs text-muted-foreground">
                  {removeTarget.reference_images.length} reference face image
                  {removeTarget.reference_images.length === 1 ? "" : "s"} enrolled
                </p>
              </div>
              <label className="flex cursor-pointer items-center gap-2 text-xs font-medium">
                <input
                  type="checkbox"
                  checked={purge}
                  onChange={(event) => setPurge(event.target.checked)}
                  className="size-4 rounded border-input accent-[var(--destructive)]"
                />
                Permanently delete reference images from disk
              </label>
              <DialogFooter>
                <Button variant="outline" onClick={() => setRemoveTarget(null)}>
                  Cancel
                </Button>
                <Button
                  variant="destructive"
                  disabled={removeMutation.isPending}
                  onClick={() =>
                    removeTarget &&
                    removeMutation.mutate({ workerId: removeTarget.worker_id, deleteImages: purge })
                  }
                >
                  {removeMutation.isPending ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : purge ? (
                    <>
                      <UserRoundX className="size-4" /> Purge Worker Profile
                    </>
                  ) : (
                    <>
                      <UserRoundCheck className="size-4" /> Deactivate Worker
                    </>
                  )}
                </Button>
              </DialogFooter>
            </div>
          ) : null}
        </DialogContent>
      </Dialog>

      {/* Uploading indicator */}
      {addImagesMutation.isPending ? (
        <div
          className="fixed bottom-4 right-4 z-50 flex items-center gap-2 rounded-lg border bg-card px-4 py-2.5 text-sm font-medium shadow-lg"
          role="status"
        >
          <Loader2 className="size-4 animate-spin text-primary" /> Enrolling new reference face images…
        </div>
      ) : null}
    </div>
  );
}
