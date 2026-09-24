# Worker Face Recognition & Verification (Siamese)

This document explains the complete worker-identity layer built around the
trained Siamese verification model `notebooks/siamesemodelv2.h5`
(from `notebooks/Facial Verification with a Siamese Network.ipynb`):
**where worker data lives, how to register a worker, how the model is
loaded, how verification works, how thresholds are calibrated, and how to
run everything.**

---

## 1. Where worker data lives

All identity data is stored under a single configurable root
(`FACE_DATA_DIR`, default `data/face_data/`):

```text
data/face_data/
├── workers.json                        # worker registry (auto-managed)
├── input_images/
│   ├── worker_001/                     # one folder per worker
│   │   ├── <uuid>.jpg                  # reference face crops
│   │   └── ...
│   └── worker_002/
│       └── ...
├── embeddings/
│   ├── worker_001.npy                  # cached 4096-d embeddings (auto)
│   └── worker_002.npy
└── calibration.json                    # calibrated thresholds (optional)
```

- `workers.json` is the source of truth: worker id, name, role, active
  flag, and the list of reference image paths. Never edit it by hand
  while the server is running (the registry writes it atomically).
- **Reference images are the crops themselves** — exactly the kind of
  250×250-ish face crops the notebook's anchor-collection flow produced.
  5–10 varied shots per worker (different angles/lighting/expression)
  gives robust verification.
- `embeddings/*.npy` is a pure performance cache (per-worker 4096-d
  embeddings). It is invalidated automatically whenever a reference
  image is added/removed; verification works with or without it.
- `calibration.json` is produced by `scripts/calibrate_threshold.py` and
  is picked up automatically at startup (see §4).

**Zero-code worker addition:** create
`data/face_data/input_images/<worker_id>/` with face jpgs inside; the app
and every CLI script auto-register that folder on startup
(`registry.auto_discover_workers()`). You can also manage workers live
over REST (see §7) without restarting anything.

---

## 2. How the model is loaded

- Model file: `notebooks/siamesemodelv2.h5` (configurable via
  `FACE_MODEL_PATH`). Built in the notebook as
  `input_img, validation_img (100×100×3) → shared 'embedding' model
  (4096-d sigmoid) → L1 distance → Dense(1, sigmoid)`.
- `src/face_recognition/backends.py::SiameseFaceBackend` loads it with:

  ```python
  tf.keras.models.load_model(
      path,
      custom_objects={"L1Dist": L1Dist,
                      "BinaryCrossentropy": tf.losses.BinaryCrossentropy},
      compile=False,
  )
  ```

  `L1Dist` is the custom layer saved inside the `.h5` (re-declared to
  match the notebook's class). TensorFlow is imported **only** inside the
  backend constructor, so the rest of the app imports and runs without TF.
- The backend is created **once** (app startup in `app/main.py::_build_face_service`)
  and shared across all streams; model calls are serialised behind a lock
  (same pattern as `HazardDetector`).
- **Graceful degradation:** if `FACE_RECOGNITION_ENABLED=false`, TF is
  missing, or the `.h5` is missing, `FACE_SERVICE` is `None`, a warning is
  logged, and the app runs exactly as before — all `/face/*` endpoints
  return HTTP 503.
- **Environment note:** the trained `.h5` is a TF-2.4-era graph. Modern
  TensorFlow (>= 2.16, Keras 3) cannot rebuild it directly; install
  `tensorflow-cpu` **plus** `tf-keras` (the official maintained Keras 2,
  both already in `requirements.txt`). The loader prefers `tf-keras`
  automatically and falls back to bundled Keras on older TF, so one
  Python environment (the app `venv`) runs the whole system. Tests use
  `StubFaceBackend` and run anywhere without TF.

---

## 3. How verification works

Per frame (inside `SafetyPipeline.process_frame`, so webcam, video, and
REST all share one code path):

```text
frame
  → HazardDetector (YOLO) → Person boxes (already filtered/confirmed)
  → FaceDetector: top 35% of each person box (head band), narrowed to
    80% width, clamped; crops < 40px skipped
  → preprocessing: RGB, resize 100×100 bilinear, /255  (notebook-exact)
  → for each registered worker:
        scores = Siamese(live, ref_1..ref_N)        # N pair scores
        detections = count(scores > detection_threshold)
        rate = detections / N
        verified_worker = rate > verification_threshold
  → pick best verified worker (highest rate, then mean pair score)
  → optional runner-up margin (FACE_MIN_MARGIN) → else Unknown
  → per-track cache for FACE_CACHE_TTL_S seconds
```

- **Multiple workers in one frame:** every person box is processed
  independently (up to `FACE_MAX_FACES_PER_FRAME`), each compared against
  **all** registered workers. Each face gets its own identity; everyone
  who fails is `Unknown/Not Verified`.
- **Multiple faces per worker set:** the decision margin prevents
  ambiguous assignments between two similar registered workers.
- Output per face (`FrameIdentity`): verified flag, worker_id/worker_name,
  rate + mean score, all per-worker match details, and a JSON-safe
  `to_dict()`. The pipeline attaches these to `FrameResult.identities`,
  includes them in the REST/websocket payloads (`identities`,
  `identity_summary`), and draws green `Name (xx%)` / red `UNKNOWN` boxes
  plus a `FACES: ...` HUD banner.
- Safety behaviour is untouched: PPE, tracking, danger zones, proximity,
  fire/smoke, risk levels, alerts, evidence all continue unchanged; the
  identity layer is additive and failure-isolated (a crash in face
  recognition can never break the safety pipeline — see
  `tests/test_pipeline_identity.py`).

### Threshold semantics (two thresholds, as in the notebook's `verify()`)

| Threshold | Meaning | Default | Env var |
|---|---|---|---|
| `detection_threshold` | per-pair score above which one (live, ref) comparison counts as a match | **0.50** | `FACE_DETECTION_THRESHOLD` |
| `verification_threshold` | fraction of the worker's references that must match | **0.60** | `FACE_VERIFICATION_THRESHOLD` |

The notebook used 0.9/0.7, but measurement on this model's own training
data shows impostor pairs score ≈ 0.0 while genuine webcam pairs score
median ≈ 0.62 (max 1.0) — so 0.9 rejected most genuine pairs. 0.5 sits in
the separation gap. **Re-calibrate on your own workers** (next section).
Note: a live crop pixel-identical to a reference scores ≈ 0.5 (the model
never saw zero-distance pairs in training); real captures always differ,
so this never matters in production.

---

## 4. Threshold calibration

```bash
# after registering at least one worker with 2+ reference images
# (ideally 2+ workers for a meaningful impostor set):
python scripts/calibrate_threshold.py
```

The script scores genuine pairs (live vs the same worker's other refs,
leave-one-out) and impostor pairs (live vs every other worker's refs),
finds the separating threshold, sweeps the verification fraction for the
best TAR−FAR, prints the full report, and writes
`data/face_data/calibration.json`. The app + CLI scripts apply it
automatically at startup; `POST /face/calibrate` re-runs it live.

Example output from this model:

```json
{
  "suggested_detection_threshold": 0.3,
  "suggested_verification_threshold": 0.3,
  "true_accept_rate_at_suggestion": 0.8333,
  "false_accept_rate_at_suggestion": 0.0,
  "genuine_min": 0.087, "genuine_mean": 0.7342, "impostor_max": 0.0,
  "n_genuine_pairs": 12, "n_impostor_pairs": 18
}
```

With `impostor_max = 0.0`, raising `FACE_DETECTION_THRESHOLD` toward
0.6–0.7 trades a little true-accept rate for a large safety margin.

---

## 5. How to register a new worker

### Option A — from saved face crops (recommended)

```bash
python scripts/register_worker.py \
    --worker-id worker_003 --name "Omar Khaled" --role "electrician" \
    --images crops/omar1.jpg crops/omar2.jpg "crops/omar_extra/*.jpg"
```

### Option B — live from the webcam

```bash
python scripts/register_worker.py \
    --worker-id worker_003 --name "Omar Khaled" --capture
# C = capture a reference crop, R = finish, Q = quit
```

### Option C — drop files, zero commands

Create `data/face_data/input_images/worker_003/` with jpgs; the next app
start (or CLI run) auto-registers the folder with worker_id = folder name.

### Option D — REST (server running)

```bash
curl -X POST "http://127.0.0.1:8000/face/workers?worker_id=worker_003&name=Omar" \
     -F "files=@crops/omar1.jpg" -F "files=@crops/omar2.jpg"
```

### Inspect / remove

```bash
python scripts/register_worker.py --list
python scripts/register_worker.py --worker-id worker_003 --remove   # deactivate
python scripts/register_worker.py --worker-id worker_003 --purge    # delete + remove files
```

Good references = 5–10 crops per worker, 250×250-ish, varied lighting /
angle / expression, face only (like the notebook's anchor flow).

---

## 6. How to run the complete system

```bash
# 1. Register workers (any option from §5)

# 2. (Recommended) calibrate thresholds on your workers
python scripts/calibrate_threshold.py

# 3a. Live verification demo only (no safety pipeline)
python scripts/face_verify_webcam.py

# 3b. Webcam with SAFETY + identities together
python scripts/live_webcam.py --face-recognition

# 3c. Offline video with SAFETY + identities
python scripts/analyze_video.py \
    --video data/videos/test1.mp4 --face-recognition

# 3d. Full web dashboard + REST API (run from the TF environment)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The dashboard payload now includes `identities` (per-face results) and
`identity_summary` (`faces_seen`, `workers_verified`, `verified_workers`,
`unknown_faces`) alongside the existing safety fields.

### REST API for face recognition

| Method & path | Purpose |
|---|---|
| `GET  /face/workers` | list workers (incl. deactivated) |
| `POST /face/workers?worker_id=&name=&role=` + `files[]` | register worker + reference images |
| `POST /face/workers/{worker_id}/images` + `files[]` | add more references |
| `DELETE /face/workers/{worker_id}?delete_images=` | deactivate / purge |
| `POST /face/verify` (file upload) | verify faces in one image |
| `POST /face/calibrate` | recalibrate thresholds from registered data |

### Configuration (env / .env)

| Variable | Default | Meaning |
|---|---|---|
| `FACE_RECOGNITION_ENABLED` | `true` | master switch (`false` = feature off) |
| `FACE_MODEL_PATH` | `notebooks/siamesemodelv2.h5` | trained Siamese model |
| `FACE_DATA_DIR` | `data/face_data` | identity data root |
| `FACE_DETECTION_THRESHOLD` | `0.50` | per-pair match threshold |
| `FACE_VERIFICATION_THRESHOLD` | `0.60` | required fraction of refs |
| `FACE_MIN_MARGIN` | `0.0` | runner-up margin (ambiguity guard) |
| `FACE_FRAME_STRIDE` | `1` | verify every Nth frame (websocket loop) |
| `FACE_CACHE_TTL_S` | `2.0` | per-track identity cache seconds |
| `FACE_MIN_FACE_SIZE_PX` | `40` | skip smaller face crops |
| `FACE_MAX_FACES_PER_FRAME` | `5` | bandwidth cap per frame |
| `FACE_MIN_PERSON_CONFIDENCE` | `0.45` | min YOLO person conf |
| `FACE_BACKEND` | `siamese` | `siamese` (real model) or `stub` (tests/CI) |

---

## 7. Module map

```text
src/face_recognition/
├── config.py         FaceRecognitionConfig (all tunables, env-overridable)
├── preprocessing.py  notebook-exact crop/resize/normalise (TF-free)
├── backends.py       SiameseFaceBackend (TF, lazy) + StubFaceBackend
├── registry.py       WorkerRegistry: workers.json, reference images, embeddings
├── face_detector.py  person box -> head-band face crops
├── service.py        FaceRecognitionService: multi-worker verify + cache
├── overlay.py        identity boxes/labels + FACES banner
└── calibration.py    genuine/impostor sweep -> calibration.json

Integration points
├── src/pipeline.py   SafetyPipeline.process_frame attaches FrameResult.identities
│                     and draws the overlay; face_frame_stride support
├── app/main.py       FACE_SERVICE singleton, /face/* REST API, payload fields
├── app/settings.py   FACE_RECOGNITION_ENABLED / FACE_BACKEND / FACE_FRAME_STRIDE
└── scripts/          register_worker.py, calibrate_threshold.py,
                      face_verify_webcam.py, live_webcam.py, analyze_video.py
```

---

## 8. Tests

```bash
venv/Scripts/python.exe -m pytest tests/test_face_recognition.py tests/test_pipeline_identity.py -q
```

34 model-free tests cover: registry CRUD/persistence/sanitisation,
auto-discovery, notebook-exact preprocessing contract, head-band face
crops, multi-worker verification, Unknown handling, ambiguity margin,
cache TTL, stride behaviour, crash isolation, JSON-safe payloads,
calibration maths, and the backend factory. The real `.h5` backend is
exercised manually via `scripts/face_verify_webcam.py` /
`scripts/calibrate_threshold.py` (needs the TF environment).
