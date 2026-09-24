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
  UserPlus,
  UserRoundCheck,
  UserRoundX,
} from "lucide-react";
import { PageHeader } from "@/components/page-header";
import { StatusBadge } from "@/components/status-badge";
import { EmptyState, ErrorState, LoadingCard, SectionCard } from "@/components/stat-card";
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
      { title: "Workers | SentinelOps" },
      {
        name: "description",
        content: "Manage registered workers and their reference face images for live verification.",
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
    <article className="panel-rise flex flex-col rounded-lg border bg-card p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <span className="grid size-10 shrink-0 place-items-center rounded-full bg-primary/10 text-primary">
            <ScanFace className="size-5" />
          </span>
          <div className="min-w-0">
            <h3 className="truncate font-display text-sm font-semibold">
              {worker.name || worker.worker_id}
            </h3>
            <p className="truncate text-xs text-muted-foreground">
              {worker.role || "Worker"} · <code className="text-[10.5px]">{worker.worker_id}</code>
            </p>
          </div>
        </div>
        <StatusBadge tone={worker.active ? "success" : "neutral"}>
          {worker.active ? "Active" : "Inactive"}
        </StatusBadge>
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-3 text-xs">
        <div className="rounded-md bg-muted/60 px-3 py-2">
          <dt className="flex items-center gap-1.5 text-muted-foreground">
            <FileImage className="size-3.5" /> Reference faces
          </dt>
          <dd className="mt-1 font-display text-lg font-bold">{worker.reference_images.length}</dd>
        </div>
        <div className="rounded-md bg-muted/60 px-3 py-2">
          <dt className="flex items-center gap-1.5 text-muted-foreground">
            <CalendarClock className="size-3.5" /> Registered
          </dt>
          <dd className="mt-1 font-semibold">{formatTimestamp(worker.created_at)}</dd>
        </div>
      </dl>

      {worker.reference_images.length > 0 ? (
        <p className="mt-3 flex items-center gap-1.5 text-[11px] text-muted-foreground">
          <Images className="size-3.5" />
          stored under data/face_data/input_images/{worker.worker_id}/
        </p>
      ) : (
        <p className="mt-3 rounded-md bg-medium/10 px-2.5 py-1.5 text-[11px] text-medium">
          No reference faces yet — this worker cannot be verified on camera.
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
          <Images className="size-3.5" /> Add faces
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
        eyebrow="Face recognition"
        title="Worker management"
        description="Registered workers are verified live on camera by the Siamese model. Each worker needs 5–10 reference face crops for reliable verification."
        actions={
          <Button onClick={() => setRegisterOpen(true)}>
            <UserPlus className="size-4" /> Register worker
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
            placeholder="Search by id, name, or role…"
            className="pl-9"
            aria-label="Search workers"
          />
        </div>
        <label className="flex cursor-pointer items-center gap-2 text-sm text-muted-foreground">
          <input
            type="checkbox"
            checked={showInactive}
            onChange={(event) => setShowInactive(event.target.checked)}
            className="size-4 rounded border-input accent-[var(--primary)]"
          />
          Show deactivated
        </label>
        <span className="ml-auto text-xs text-muted-foreground">
          {workers.length} worker{workers.length === 1 ? "" : "s"} ·{" "}
          {workers.filter((worker) => worker.active).length} active
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
          message="The face-recognition service is unavailable (TensorFlow or the Siamese model is missing on the server). Worker management requires it."
          onRetry={() => void workersQuery.refetch()}
        />
      ) : filtered.length === 0 ? (
        <EmptyState
          icon={ScanFace}
          title={search ? "No workers match your search" : "No workers registered yet"}
          description={
            search
              ? "Try a different id, name, or role."
              : "Register your first worker with 5–10 frontal face crops. They will immediately be verified on the live monitor."
          }
          action={
            search ? undefined : (
              <Button onClick={() => setRegisterOpen(true)}>
                <UserPlus className="size-4" /> Register worker
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
            <DialogTitle>Register a worker</DialogTitle>
            <DialogDescription>
              The worker id becomes the folder{" "}
              <code className="text-[11px]">data/face_data/input_images/&lt;id&gt;/</code>.
              Reference faces are saved there and used for live verification.
            </DialogDescription>
          </DialogHeader>
          <form onSubmit={handleRegisterSubmit} className="space-y-4">
            <div className="grid gap-2">
              <Label htmlFor="worker-id">Worker ID *</Label>
              <Input
                id="worker-id"
                value={form.workerId}
                onChange={(event) =>
                  setForm((current) => ({ ...current, workerId: event.target.value }))
                }
                placeholder="worker_001"
                autoComplete="off"
                required
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="worker-name">Full name</Label>
              <Input
                id="worker-name"
                value={form.name}
                onChange={(event) =>
                  setForm((current) => ({ ...current, name: event.target.value }))
                }
                placeholder="Ahmed Ali"
                autoComplete="off"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="worker-role">Role</Label>
              <Input
                id="worker-role"
                value={form.role}
                onChange={(event) =>
                  setForm((current) => ({ ...current, role: event.target.value }))
                }
                placeholder="Crane operator"
                autoComplete="off"
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="worker-faces">Reference face images</Label>
              <Input
                id="worker-faces"
                type="file"
                accept="image/jpeg,image/png,image/webp"
                multiple
                onChange={(event) => setFiles(Array.from(event.target.files ?? []))}
              />
              <p className="text-[11px] text-muted-foreground">
                {files.length > 0
                  ? `${files.length} image${files.length === 1 ? "" : "s"} selected`
                  : "Optional now — you can add faces later. Frontal crops, varied lighting."}
              </p>
            </div>
            {formError ? (
              <p
                role="alert"
                className="rounded-md bg-destructive/10 px-3 py-2 text-xs text-destructive"
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
                    <UserPlus className="size-4" /> Register
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
            <DialogTitle>Remove worker</DialogTitle>
            <DialogDescription>
              Deactivating keeps the data but the worker will no longer be verified on camera.
              Purging permanently deletes their reference images.
            </DialogDescription>
          </DialogHeader>
          {removeTarget ? (
            <div className="space-y-3">
              <div className="rounded-md bg-muted/60 px-3 py-2 text-sm">
                <p className="font-semibold">{removeTarget.name || removeTarget.worker_id}</p>
                <p className="text-xs text-muted-foreground">
                  {removeTarget.reference_images.length} reference image
                  {removeTarget.reference_images.length === 1 ? "" : "s"} stored
                </p>
              </div>
              <label className="flex cursor-pointer items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={purge}
                  onChange={(event) => setPurge(event.target.checked)}
                  className="size-4 rounded border-input accent-[var(--destructive)]"
                />
                Also delete reference images permanently
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
                      <UserRoundX className="size-4" /> Purge worker
                    </>
                  ) : (
                    <>
                      <UserRoundCheck className="size-4" /> Deactivate
                    </>
                  )}
                </Button>
              </DialogFooter>
            </div>
          ) : null}
        </DialogContent>
      </Dialog>

      {/* Hidden busy indicator for inline image adds */}
      {addImagesMutation.isPending ? (
        <div
          className="fixed bottom-4 right-4 z-50 flex items-center gap-2 rounded-lg border bg-card px-4 py-2.5 text-sm shadow-lg"
          role="status"
        >
          <Loader2 className="size-4 animate-spin text-primary" /> Uploading reference faces…
        </div>
      ) : null}
    </div>
  );
}
