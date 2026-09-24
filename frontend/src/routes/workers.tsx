import { createFileRoute } from "@tanstack/react-router";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useRef, useState, type ChangeEvent, type FormEvent } from "react";
import {
  CalendarClock,
  CheckCircle2,
  FileImage,
  Images,
  Loader2,
  RotateCcw,
  ScanFace,
  Search,
  Sliders,
  Trash2,
  UploadCloud,
  UserCheck,
  UserPlus,
  UserRoundCheck,
  UserRoundX,
} from "lucide-react";
import { PageHeader } from "@/components/page-header";
import { IdentityBadge, StatusBadge } from "@/components/status-badge";
import { EmptyState, ErrorState, LoadingCard, SectionCard } from "@/components/stat-card";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
import {
  addWorkerImages,
  calibrateThresholds,
  listWorkers,
  registerWorker,
  removeWorker,
  verifyFaceImage,
} from "@/lib/api";
import type { CalibrationResult, FaceVerifyResponse, WorkerRecord } from "@/lib/types";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/workers")({
  head: () => ({
    meta: [
      { title: "Workers & Verification | SentinelOps" },
      {
        name: "description",
        content: "Manage registered workers, test face verification, and calibrate Siamese recognition thresholds.",
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
        <p className="mt-3 flex items-center gap-1.5 text-[11px] text-muted-foreground truncate">
          <Images className="size-3.5 shrink-0" />
          {worker.reference_images.length} reference crops attached
        </p>
      ) : (
        <p className="mt-3 rounded-md bg-medium/10 px-2.5 py-1.5 text-[11px] text-medium">
          No reference faces yet — worker cannot be verified on camera.
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

  // Face Verification Tool State
  const [verifyFile, setVerifyFile] = useState<File | null>(null);
  const [verifyResult, setVerifyResult] = useState<FaceVerifyResponse | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [verifyError, setVerifyError] = useState<string | null>(null);

  // Calibration Tool State
  const [calibrationData, setCalibrationData] = useState<CalibrationResult | null>(null);
  const [calibrationOpen, setCalibrationOpen] = useState(false);

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

  const calibrateMutation = useMutation({
    mutationFn: calibrateThresholds,
    onSuccess: (data) => {
      if (data.calibration) {
        setCalibrationData(data.calibration);
        setCalibrationOpen(true);
      }
    },
  });

  const handleVerifySubmit = async (file: File) => {
    setVerifyFile(file);
    setIsVerifying(true);
    setVerifyError(null);
    setVerifyResult(null);
    try {
      const res = await verifyFaceImage(file);
      setVerifyResult(res);
    } catch (err: unknown) {
      setVerifyError(err instanceof Error ? err.message : "Verification failed");
    } finally {
      setIsVerifying(false);
    }
  };

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
        eyebrow="Face recognition & registry"
        title="Worker identity management"
        description="Registered workers are verified live on camera by the Siamese deep learning model. Manage reference faces, test face identification, and recalibrate model decision thresholds."
        actions={
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              disabled={calibrateMutation.isPending}
              onClick={() => calibrateMutation.mutate()}
            >
              {calibrateMutation.isPending ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Sliders className="size-4" />
              )}
              Calibrate Thresholds
            </Button>
            <Button onClick={() => setRegisterOpen(true)}>
              <UserPlus className="size-4" /> Register worker
            </Button>
          </div>
        }
      />

      {/* Face Verification Test Card */}
      <SectionCard
        title="Live Face Verification Tester"
        description="Upload a worker photo to test the Siamese face identification service against registered workers."
        className="mb-6"
      >
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <DropzoneSmall
              label="Upload test photo for face verification"
              sublabel="Select an image containing worker faces"
              onFileSelect={handleVerifySubmit}
            />
          </div>

          <div>
            {isVerifying ? (
              <div className="py-8 text-center text-xs text-muted-foreground">
                <Loader2 className="mx-auto size-6 animate-spin text-primary" />
                <p className="mt-2 font-semibold">Running face detection & Siamese matching…</p>
              </div>
            ) : verifyError ? (
              <div className="rounded-md bg-destructive/10 p-3 text-xs text-destructive">
                <p className="font-semibold">Verification Error:</p>
                <p className="mt-0.5">{verifyError}</p>
              </div>
            ) : verifyResult ? (
              <div className="space-y-2 text-xs">
                <div className="flex items-center justify-between border-b pb-2">
                  <span className="font-semibold">Detected Faces: {verifyResult.faces?.length ?? 0}</span>
                  <span className="text-safe font-semibold">
                    Verified: {verifyResult.verified_workers?.length ?? 0}
                  </span>
                </div>

                {verifyResult.faces && verifyResult.faces.length > 0 ? (
                  <ul className="space-y-2">
                    {verifyResult.faces.map((face, idx) => (
                      <li key={idx} className="flex items-center justify-between rounded border p-2 bg-card">
                        <div className="min-w-0">
                          <p className="font-semibold truncate">
                            {face.verified ? face.worker_name || face.worker_id : "Unrecognized Face"}
                          </p>
                          <p className="text-[10px] text-muted-foreground">
                            Match Score: {(face.score * 100).toFixed(1)}%
                          </p>
                        </div>
                        <IdentityBadge verified={face.verified} />
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-muted-foreground">No faces detected in this image.</p>
                )}
              </div>
            ) : (
              <p className="py-6 text-center text-xs text-muted-foreground">
                Upload a photo on the left to see instant face identification results.
              </p>
            )}
          </div>
        </div>
      </SectionCard>

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

      {/* Calibration Dialog */}
      <Dialog open={calibrationOpen} onOpenChange={setCalibrationOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Sliders className="size-4 text-primary" /> Threshold Calibration Complete
            </DialogTitle>
            <DialogDescription>
              Recalibrated verification thresholds based on current registered worker reference pairs.
            </DialogDescription>
          </DialogHeader>
          {calibrationData ? (
            <div className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-2">
                <div className="rounded-lg bg-muted p-2.5">
                  <p className="text-[10px] uppercase font-bold text-muted-foreground">Detection Threshold</p>
                  <p className="text-base font-bold font-display text-primary">
                    {calibrationData.suggested_detection_threshold.toFixed(2)}
                  </p>
                </div>
                <div className="rounded-lg bg-muted p-2.5">
                  <p className="text-[10px] uppercase font-bold text-muted-foreground">Verification Threshold</p>
                  <p className="text-base font-bold font-display text-primary">
                    {calibrationData.suggested_verification_threshold.toFixed(2)}
                  </p>
                </div>
              </div>

              <div className="rounded-lg border p-3 space-y-1">
                <p className="font-semibold text-foreground">Performance Metrics:</p>
                <p>True Accept Rate (TAR): <strong>{(calibrationData.true_accept_rate_at_suggestion * 100).toFixed(1)}%</strong></p>
                <p>False Accept Rate (FAR): <strong>{(calibrationData.false_accept_rate_at_suggestion * 100).toFixed(1)}%</strong></p>
                <p>Genuine pairs checked: {calibrationData.n_genuine_pairs}</p>
                <p>Impostor pairs checked: {calibrationData.n_impostor_pairs}</p>
              </div>
            </div>
          ) : null}
          <DialogFooter>
            <Button onClick={() => setCalibrationOpen(false)}>Done</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Register dialog */}
      <Dialog open={registerOpen} onOpenChange={setRegisterOpen}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>Register a worker</DialogTitle>
            <DialogDescription>
              Reference faces are saved and used for live verification.
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
              Deactivating keeps data; purging permanently deletes reference images.
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
    </div>
  );
}

function DropzoneSmall({
  label,
  sublabel,
  onFileSelect,
}: {
  label: string;
  sublabel: string;
  onFileSelect: (file: File) => void;
}) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (files.length > 0) onFileSelect(files[0]!);
  };

  return (
    <div
      onClick={() => fileInputRef.current?.click()}
      className="flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-4 text-center transition-colors border-border hover:border-primary/50 hover:bg-accent/50"
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        hidden
        onChange={handleChange}
        aria-label={label}
      />
      <UploadCloud className="size-5 text-primary" />
      <p className="mt-1 font-display text-xs font-semibold">{label}</p>
      <p className="text-[10px] text-muted-foreground">{sublabel}</p>
    </div>
  );
}
