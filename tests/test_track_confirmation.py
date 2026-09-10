"""Unit tests for src.hazard.track_confirmation (debounce + static-object suppressors)."""

from src.hazard.track_confirmation import TrackConfirmationTracker

FRAME = (640, 640)


def _det(track_id, class_name, bbox, confidence=0.9):
    return {
        "track_id": track_id,
        "class_name": class_name,
        "confidence": confidence,
        "bbox": list(bbox),
    }


def _feed(tracker, det, frames):
    """Feed the same detection for `frames` frames, return last output."""
    out = []
    for _ in range(frames):
        out = tracker.update([dict(det)], frame_shape=FRAME)
    return out


# ------------------------------------------------------------------
# passthrough + debounce
# ------------------------------------------------------------------

def test_untracked_detection_passes_through():
    tracker = TrackConfirmationTracker(min_hits=3)
    det = _det(None, "Person", [100, 400, 160, 600])
    assert tracker.update([det], frame_shape=FRAME) == [det]


def test_track_needs_min_hits_before_confirmation():
    tracker = TrackConfirmationTracker(min_hits=3)
    det = _det(1, "Person", [100, 400, 160, 600])
    assert tracker.update([det], frame_shape=FRAME) == []   # hit 1
    assert tracker.update([det], frame_shape=FRAME) == []   # hit 2
    assert tracker.update([det], frame_shape=FRAME) == [det]  # hit 3 -> confirmed


def test_track_ages_out_after_max_missed_frames():
    tracker = TrackConfirmationTracker(min_hits=1, max_missed_frames=2)
    det = _det(1, "Person", [100, 400, 160, 600])
    tracker.update([det], frame_shape=FRAME)
    assert tracker.hits_for(1) == 1
    # Miss it three times (max_missed_frames=2) -> dropped.
    for _ in range(3):
        tracker.update([], frame_shape=FRAME)
    assert tracker.hits_for(1) == 0


# ------------------------------------------------------------------
# static PERSON suppressor (pole / stake mislabelled as Person)
# ------------------------------------------------------------------

def test_static_low_confidence_person_is_suppressed():
    tracker = TrackConfirmationTracker(
        min_hits=2, static_min_frames=5, static_motion_threshold_px=3.0,
        static_max_avg_confidence=0.55,
    )
    det = _det(1, "Person", [100, 400, 160, 600], confidence=0.40)
    # Well past the static window -> dropped.
    assert _feed(tracker, det, 8) == []


def test_static_high_confidence_person_is_kept():
    # A clearly-visible worker who happens to stand still must NOT be dropped.
    tracker = TrackConfirmationTracker(
        min_hits=2, static_min_frames=5, static_max_avg_confidence=0.55,
    )
    det = _det(1, "Person", [100, 400, 160, 600], confidence=0.90)
    assert len(_feed(tracker, det, 8)) == 1


def test_moving_person_is_kept():
    tracker = TrackConfirmationTracker(
        min_hits=2, static_min_frames=5, static_max_avg_confidence=0.99,
    )
    out = None
    for i in range(8):
        det = _det(1, "Person", [100 + i * 20, 400, 160 + i * 20, 600], confidence=0.40)
        out = tracker.update([det], frame_shape=FRAME)
    assert len(out) == 1


def test_static_person_filter_can_be_disabled():
    tracker = TrackConfirmationTracker(
        min_hits=2, filter_static_persons=False, static_min_frames=5,
    )
    det = _det(1, "Person", [100, 400, 160, 600], confidence=0.40)
    assert len(_feed(tracker, det, 8)) == 1


# ------------------------------------------------------------------
# static MACHINERY suppressor (building mislabelled as machinery)
# ------------------------------------------------------------------

def test_static_machinery_is_suppressed_regardless_of_confidence():
    tracker = TrackConfirmationTracker(
        min_hits=2, static_machinery_min_frames=5,
        static_machinery_motion_threshold_px=2.5,
    )
    # High confidence (0.8) - exactly the house->machinery FP signature.
    det = _det(1, "machinery", [300, 200, 500, 400], confidence=0.80)
    assert _feed(tracker, det, 8) == []


def test_moving_machinery_is_kept():
    tracker = TrackConfirmationTracker(
        min_hits=2, static_machinery_min_frames=5,
    )
    out = None
    for i in range(8):
        det = _det(1, "machinery", [300 + i * 20, 200, 500 + i * 20, 400], confidence=0.80)
        out = tracker.update([det], frame_shape=FRAME)
    assert len(out) == 1


def test_static_machinery_filter_can_be_disabled():
    tracker = TrackConfirmationTracker(
        min_hits=2, filter_static_machinery=False, static_machinery_min_frames=5,
    )
    det = _det(1, "machinery", [300, 200, 500, 400], confidence=0.80)
    assert len(_feed(tracker, det, 8)) == 1
