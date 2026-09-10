"""
Central safety-analysis engine.

Pipeline:

    YOLO hazard detections (persons/PPE/cones/machinery/poles/vehicles)
          + YOLO fire/smoke detections
          v
    Dangerous zones (optionally stabilized across frames)
          v
    Person-zone analysis
          v
    Person-machine distance
          v
    PPE analysis
          v
    Utility-pole proximity
          v
    Fire / smoke analysis
          v
    Safety rules
          v
    Risk level

CHANGELOG (vs original)
------------------------
- analyze() now accepts an optional `fire_detections` list (from
  FireDetector) and folds FIRE/SMOKE violations into the same result,
  so the risk level genuinely reflects "is there a fire" and not just
  PPE/zone/proximity hazards.
- SafetyEngine can optionally be constructed with a DangerZoneTracker
  to give danger zones a persistent zone_id across frames of the same
  video/stream instead of recomputing untracked zones every frame.
  This is opt-in and stateful - use ONE SafetyEngine (or at least one
  zone_tracker) PER video/camera stream, not a shared global instance,
  or zones from unrelated streams will get matched against each other.
"""

from dataclasses import replace
from typing import Any, Dict, List, Optional

from src.safety.geometry import (
    build_danger_zones,
    find_people_inside_zones,
    find_machinery_near_poles,
    DangerZoneTracker,
)

from src.safety.distance_calculator import (
    calculate_person_machine_distances,
)

from src.safety.rules import (
    SafetyConfig,
    get_ppe_violations,
    get_zone_violations,
    get_machine_distance_violations,
    get_pole_proximity_violations,
    get_fire_violations,
    calculate_risk_level,
    scaled_pixel_threshold,
)


class SafetyEngine:
    """
    Main safety-analysis engine.

    The engine does NOT perform YOLO inference. It receives already-run
    detections and converts them into safety violations + a risk level.
    """

    def __init__(
        self,
        config: Optional[SafetyConfig] = None,
        zone_tracker: Optional[DangerZoneTracker] = None,
    ):
        self.config = config if config is not None else SafetyConfig()

        # None is a valid, supported choice: zones are then recomputed
        # from scratch every call with no persistent identity, which is
        # fine for single-image analysis.
        self.zone_tracker = zone_tracker

    # ========================================================
    # MAIN FRAME ANALYSIS
    # ========================================================

    def analyze(
        self,
        detections: List,
        persons: List[Dict],
        machines: List[Dict],
        fire_detections: Optional[List[Dict]] = None,
        frame_width: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Analyze one frame.

        Parameters
        ----------
        detections:
            Full hazard-model detections for this frame (persons, PPE,
            cones, machinery, poles, vehicles combined) - typically
            HazardDetector.track(frame).

        persons:
            Tracked persons, e.g. HazardDetector.extract_persons(detections).

        machines:
            Machinery/vehicle detections, e.g.
            HazardDetector.extract_machines(detections).

        fire_detections:
            Optional normalized fire/smoke detections, e.g.
            FireDetector.predict(frame). Pass None or [] if not running
            fire detection on this frame.

        frame_width:
            Current frame width in pixels. When set, proximity
            thresholds and zone buffers scale from the 1280px
            calibration width so webcam and 4K video share one config.
        """

        if fire_detections is None:
            fire_detections = []

        config = self.config
        if config.scale_thresholds_to_frame and frame_width:
            scale = float(frame_width) / float(config.reference_frame_width)
            config = replace(
                config,
                machine_distance_threshold=scaled_pixel_threshold(
                    config.machine_distance_threshold,
                    frame_width,
                    config.reference_frame_width,
                ),
                pole_distance_threshold=scaled_pixel_threshold(
                    config.pole_distance_threshold,
                    frame_width,
                    config.reference_frame_width,
                ),
                zone_buffer_px=config.zone_buffer_px * scale,
                zone_min_area_px=config.zone_min_area_px * (scale ** 2),
            )

        # ====================================================
        # 1. DANGEROUS ZONES (raw, then optionally stabilized)
        # ====================================================

        raw_zones = build_danger_zones(
            detections=detections,
            min_cluster_size=config.cone_min_cluster_size,
            min_samples=config.cone_min_samples,
            buffer_px=config.zone_buffer_px,
            min_area_px=config.zone_min_area_px,
        )

        zone_id_by_index: Dict[int, int] = {}

        if self.zone_tracker is not None:
            stabilized = self.zone_tracker.update(raw_zones)
            # Only zones still "active" (re-detected this frame, or missed
            # within the short grace window that bridges cone flicker)
            # count as live hazards. Stale zones whose cones were genuinely
            # removed keep their id inside the tracker for continuity but
            # stop trapping people / being drawn as danger.
            active_zones = [
                z for z in stabilized
                if z.get("missed", 0) <= config.zone_presence_grace_frames
            ]
            danger_zone_polygons = [z["polygon"] for z in active_zones]
            # find_people_inside_zones matches by position in the list
            # it's given, so map that position back to the persistent id.
            zone_id_by_index = {
                index: z["zone_id"] for index, z in enumerate(active_zones)
            }
            danger_zones_meta = active_zones
        else:
            danger_zone_polygons = raw_zones
            danger_zones_meta = [
                {"zone_id": index + 1, "polygon": polygon, "age": 1, "missed": 0}
                for index, polygon in enumerate(raw_zones)
            ]

        # ====================================================
        # 2. PEOPLE INSIDE DANGER ZONES
        # ====================================================

        people_inside_zones = find_people_inside_zones(
            persons=persons,
            polygons=danger_zone_polygons,
        )

        # Re-map to persistent zone ids when a tracker is in use, so
        # alerts/history don't churn ids every frame.
        if zone_id_by_index:
            for item in people_inside_zones:
                item["zone_id"] = zone_id_by_index.get(
                    item["zone_index"], item["zone_id"]
                )

        # ====================================================
        # 3. PERSON <-> MACHINE DISTANCE
        # ====================================================

        distance_results = calculate_person_machine_distances(
            persons=persons,
            machines=machines,
            assumed_person_height_m=config.assumed_person_height_m,
        )

        # ====================================================
        # 4. PPE VIOLATIONS
        # ====================================================

        ppe_violations = get_ppe_violations(
            persons=persons,
            detections=detections,
            config=config,
        )

        # ====================================================
        # 5. DANGER-ZONE VIOLATIONS
        # ====================================================

        zone_violations = get_zone_violations(people_inside_zones)

        # ====================================================
        # 6. MACHINE PROXIMITY
        # ====================================================

        machine_violations = get_machine_distance_violations(
            distance_results=distance_results,
            config=config,
        )

        # ====================================================
        # 7. UTILITY-POLE PROXIMITY
        # ====================================================

        pole_results = find_machinery_near_poles(
            detections=detections,
            threshold_pixels=config.pole_distance_threshold,
        )

        pole_violations = get_pole_proximity_violations(
            pole_results=pole_results,
            config=config,
        )

        # ====================================================
        # 8. FIRE / SMOKE
        # ====================================================

        fire_violations = get_fire_violations(
            fire_detections=fire_detections,
            config=config,
        )

        # ====================================================
        # 9. COMBINE VIOLATIONS
        # ====================================================

        violations = (
            ppe_violations
            + zone_violations
            + machine_violations
            + pole_violations
            + fire_violations
        )

        # ====================================================
        # 10. RISK LEVEL
        # ====================================================

        risk_level = calculate_risk_level(violations)

        # ====================================================
        # 11. VIOLATION COUNTS
        # ====================================================

        violation_counts: Dict[str, int] = {}

        for violation in violations:
            violation_type = violation.get("type", "UNKNOWN")
            violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1

        # ====================================================
        # 12. RETURN RESULT
        # ====================================================

        return {
            "risk_level": risk_level,
            "violations": violations,
            "violation_count": len(violations),
            "violation_counts": violation_counts,
            "danger_zones": danger_zones_meta,
            "people_inside_zones": people_inside_zones,
            "distance_results": distance_results,
            "pole_results": pole_results,
            "fire_detections": fire_detections,
            "proximity_threshold_px": config.machine_distance_threshold,
            "use_perspective_distance": config.use_perspective_distance,
            "machine_distance_threshold_m": config.machine_distance_threshold_m,
            "statistics": {
                "persons": len(persons),
                "machines": len(machines),
                "danger_zones": len(danger_zone_polygons),
                "people_in_danger_zones": len(people_inside_zones),
                "violations": len(violations),
                "fire_detections": len(fire_detections),
            },
        }


# ============================================================
# SIMPLE FUNCTION API
# ============================================================

def analyze_frame(
    detections: List,
    persons: List[Dict],
    machines: List[Dict],
    fire_detections: Optional[List[Dict]] = None,
    config: Optional[SafetyConfig] = None,
    zone_tracker: Optional[DangerZoneTracker] = None,
    frame_width: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convenience API for one-off / stateless analysis.

    For multi-frame video or a live camera stream, prefer constructing
    one SafetyEngine (with its own zone_tracker) and reusing it across
    frames, rather than calling this function repeatedly - each call
    here with a fresh zone_tracker=None means zones won't be stabilized
    across frames.
    """

    engine = SafetyEngine(config=config, zone_tracker=zone_tracker)

    return engine.analyze(
        detections=detections,
        persons=persons,
        machines=machines,
        fire_detections=fire_detections,
        frame_width=frame_width,
    )
