# Detection Limitations: What Code Can Fix vs. What Needs the Model

This document separates the failure modes of the safety pipeline into two
buckets:

- **Code-fixable** — solvable (or already solved) with filtering, geometry,
  temporal logic, or thresholds. No retraining required.
- **Model / dataset-bound** — the network itself produces a wrong or missing
  answer. Code can only *mitigate* these; the real fix is more/better training
  data (hard negatives, class rebalancing, resolution-aware augmentation).

Every claim below points at the module that implements the mitigation so the
boundary stays honest and auditable.

---

## 1. The pipeline at a glance

```
frame
  -> HazardDetector.track()/predict()                 (YOLO forward pass)
  -> filter_detections (conf / size / aspect / ROI)    [stateless]   src/hazard/detection_filter.py
  -> TrackConfirmationTracker (debounce + static obj)  [stateful]    src/hazard/track_confirmation.py
  -> FireDetector.predict()                            (YOLO forward pass)
  -> FireConfirmationTracker (fire & smoke debounced)  [stateful]    src/fire/fire_confirmation.py
  -> extract_persons / extract_machines                              src/hazard/hazard_detector.py
  -> SafetyEngine.analyze (zones, distance, PPE, proximity, fire)    src/safety/safety_engine.py
  -> draw_safety_overlay                                             src/safety/overlay.py
```

One shared implementation (`src/pipeline.py :: SafetyPipeline`) drives the REST
image endpoint, the REST video endpoint, the websocket camera feed, the offline
video CLI, and the live webcam CLI. **A code fix lands in all five entry points
at once.** A model fix lands wherever the weights are used — also all five, but
only after retraining.

---

## 2. Code-fixable problems (and where they are fixed)

| Problem | Root cause | Fix in code | Module |
|---|---|---|---|
| Single-frame flicker detections reach the engine | ByteTrack assigns a persistent id even to a 1-frame false positive | Require `min_hits` consecutive sightings (debounce) before a track counts | `track_confirmation.py` |
| Zone ids churn frame-to-frame | `build_danger_zones` re-runs HDBSCAN + hull every frame | `DangerZoneTracker` matches zones by polygon IoU and keeps a stable `zone_id` | `geometry.py` |
| Zone traps a worker after cones are removed | Tracker keeps an id alive for continuity | `zone_presence_grace_frames` gates whether a *stale* zone still counts as a live hazard | `safety_engine.py`, `rules.py` |
| Tiny 15px smudge classified as Person | No per-class size floor; a 30px threshold lets micro-debris through at 1080p/4K | Per-class min size, **scaled by frame resolution** and by vertical perspective | `detection_filter.py :: passes_size` |
| Thin pole / wire classified as Person | Shape never checked | Person aspect-ratio gate (needle boxes rejected); stricter bounds below 0.65 confidence | `detection_filter.py :: person_aspect_ratio_ok` |
| Partial object at frame edge accepted too easily | Cropped peripheral boxes are lower quality | Edge-of-frame confidence bonus (must clear a higher bar) | `detection_filter.py :: passes_confidence` |
| NO-HARDHAT attributed to the wrong person / to someone's feet | Old check overlapped the PPE box against the *whole* person bbox | Anatomical matching: hardhat/mask vs. **head region**, vest vs. **torso region**; each PPE box assigned to at most one (best-overlap) person | `rules.py :: get_ppe_violations` |
| Pixel proximity unreliable across depth | Two far objects 20px apart look "close"; two near objects 200px apart look "far" | Calibration-free perspective: person bbox height -> `px_per_m` -> gap in **metres** (`machine_distance_threshold_m`); pixel gap kept as fallback for tiny/partial person boxes | `distance_calculator.py :: calculate_perspective_distance`, `rules.py :: get_machine_distance_violations` |
| Center-to-center distance overstates gap to large machines | Machine centroid can sit far inside a huge box while a worker is at the tracks | Edge-to-edge `bbox_separation` used for proximity | `geometry.py :: bbox_separation` |
| Glare / tools / sparks go CRITICAL instantly | Raw fire detections trusted immediately | Independent fire & smoke confirmation (type-specific confidence + area + consecutive-frame persistence, IoU-matched) | `fire_confirmation.py` |
| Faint smoke killed by fire-tuned thresholds | Fire and smoke share one model but behave differently | Separate, looser smoke gate (lower conf/area, shorter streak, lower IoU match) | `fire_confirmation.py` |
| Risk level downgraded by violation *count* | Old logic branched on count before severity | Risk derived from each violation's own `severity`, with documented compound upgrades (zone+machine, >=2 serious) | `rules.py :: calculate_risk_level` |
| "Missing mask" was never a distinct, LOW risk | NO_MASK not wired through PPE rules | NO_MASK branch added -> `LOW` severity, making the LOW risk level reachable | `rules.py :: get_ppe_violations` |
| One malformed bbox crashes the whole frame | Blind 4-value unpacking | Validation + skip-and-log for bad boxes throughout | `distance_calculator.py`, `geometry.py` |
| Static background region (sky/houses/road) generates detections on a **fixed** camera | Everything in frame is considered | Opt-in ROI polygon: detections whose bottom-center falls outside are dropped | `detection_filter.py :: build_rectangular_roi`, `passes_roi` |

---

## 3. Model / dataset-bound problems (code can only mitigate)

These are cases where the network's *output is semantically wrong*. No amount of
filtering makes the model "know" the right answer — filtering can only stop a
known-bad pattern from reaching the engine, at the cost of some real detections.

### 3.1 House / building / wall classified as `machinery`

- **Why code can't fix it:** the model emits `machinery` at **high confidence
  (0.68–0.82)**. Confidence and size gates in `detection_filter.py` cannot
  separate it from a real excavator — both are large and confident.
- **Current mitigation (temporal):** a building is pixel-perfectly static for
  the whole clip, so `TrackConfirmationTracker`'s static-machinery suppressor
  drops a `machinery`/`vehicle` track that stays essentially immobile for
  `static_machinery_min_frames` (default 90 ≈ 3s). See `track_confirmation.py`.
- **Mitigation cost (false negatives):** a genuinely **parked** excavator or a
  truck stopped for several seconds is also suppressed once it crosses the
  window. Disable with `filter_static_machinery=False`
  (`--disable-static-machinery-filter` in the CLIs) if your site has
  long-stationary plant you must keep alerting on.
- **Real fix (dataset):** hard-negative fine-tuning. Collect frames where a
  building is mislabelled, annotate the building/wall with **no** machinery box,
  and retrain. This is the only thing that removes the error at the source.

### 3.2 Ground stake / pole / tripod classified as `Person`

- **Why code can't fully fix it:** when such an object is tall enough and
  confident enough, size/aspect gates pass it.
- **Current mitigation:** static-person suppressor — drops a `Person` track only
  when it has been immobile for `static_min_frames` (default 45 ≈ 1.5s) **and**
  its average confidence is below `static_max_avg_confidence` (default 0.55).
  The confidence gate is what protects real (even momentarily still) workers.
- **Mitigation cost:** a worker who stands perfectly still for >1.5s *and* is
  detected in the low-confidence band could be dropped. The window and
  confidence bar are deliberately tuned to make this rare.
- **Real fix (dataset):** label the offending static objects correctly (or as
  background) and retrain; add pole/stake hard negatives.

### 3.3 Distant / small workers and PPE missed entirely (false negatives)

- **Why code can't fix it:** if the model never emits a box, no downstream logic
  can recover it. Raising `imgsz` helps small objects but costs FPS.
- **Partial lever (not a fix):** run higher inference resolution for
  high-altitude / drone footage (`--hazard-imgsz 960` or `1280`). This is an
  inference-time tradeoff, not a model improvement.
- **Real fix (dataset):** train/ augment with small-object and high-altitude
  examples; ensure PPE classes are represented at distance.

### 3.4 Smoke missed at range (plume vanishes after the 640 resize)

- **Why code can't fix it:** the fire/smoke model must produce the candidate
  first. The confirmation gate can only accept/reject what exists.
- **Partial lever:** raise `--fire-imgsz` (e.g. 960) so distant plumes survive
  the resize; use `--disable-fire-confirmation` + `--draw-raw-fire` to diagnose
  whether the *model* saw it (dashed cyan box) versus the *gate* dropped it.
- **Real fix (dataset):** more diverse smoke/plume training imagery.

### 3.5 PPE occlusion / angle ambiguity

- **Why we deliberately don't "fix" it in code:** `get_ppe_violations` only
  fires on **explicit** `NO-HARDHAT` / `NO-SAFETY-VEST` / `NO-MASK` detections.
  We intentionally do **not** infer a violation from the *absence* of a positive
  Hardhat/Vest box — occlusion and camera angle make that far too
  false-positive-prone. Correctly seeing "no hardhat" under occlusion is a model
  capability, not a rule.

---

## 4. Concurrency boundary (architecture, not model)

- **ByteTrack state lives inside the shared `HazardDetector`.** A single tracked
  stream (one video, one camera) is correct. Two *simultaneous* tracked streams
  sharing one detector will interleave their tracker state and corrupt ids.
- Model forward passes are serialised behind an internal `threading.RLock`, so
  concurrent calls are **memory-safe** but not independently tracked.
- **Guidance:** for multi-stream deployments, inject a dedicated
  `HazardDetector` per stream into each `SafetyPipeline`. The per-stream
  *stateful* components (zone tracker, track/fire confirmation) are already
  owned by each `SafetyPipeline` instance — never share a pipeline across
  unrelated streams.

---

## 5. How to tell which bucket a new problem falls in

1. Run the offline CLI with diagnostics:
   `python scripts/analyze_video.py --video <clip> --draw-raw-fire --disable-fire-confirmation`
2. **If the model emitted a box** (you see it raw, even dashed) but it was
   wrong/dropped -> usually **code-fixable** (filter, gate, threshold, geometry).
3. **If the model never emitted a box**, or emitted a confident *wrong class*
   (house->machinery) -> **model/dataset-bound**; code can only mitigate.
4. Check the CSV summary (`*_safety_summary.csv`): a `risk_level` of
   CRITICAL/HIGH with no visible danger means a threshold needs tuning
   (code); a hazard you can see with `risk_level` SAFE and no raw detection
   means the model missed it (dataset).

---

## 6. Quick reference

| Concern | Tunable knob | Default | Type |
|---|---|---|---|
| Track debounce | `TrackConfirmationTracker.min_hits` | 3 | code |
| Static person window | `static_min_frames` | 45 | code (mitigation) |
| Static person conf bar | `static_max_avg_confidence` | 0.55 | code (mitigation) |
| Static machinery window | `static_machinery_min_frames` | 90 | code (mitigation) |
| Person-machine proximity | `machine_distance_threshold_m` | 2.5 m | code |
| Assumed worker height | `assumed_person_height_m` | 1.7 m | code |
| PPE anatomical overlap | `ppe_overlap_threshold` | 0.40 | code |
| Zone grace window | `zone_presence_grace_frames` | 2 | code |
| Fire persistence | `min_consecutive_frames` | 5 | code |
| Smoke persistence | `smoke_min_consecutive_frames` | 2 | code |
| Small-object recall | `--hazard-imgsz` / `--fire-imgsz` | 640 | inference tradeoff |
| Wrong-class FPs (house->machinery) | hard-negative retraining | — | **dataset** |
| Missed distant workers/smoke | dataset augmentation | — | **dataset** |
