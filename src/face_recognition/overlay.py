"""
Identity overlay for the safety pipeline.

draw_identity_annotations(frame, identities, options) paints one label
per verified/unknown face on top of the (already hazard-annotated)
frame:

    [WORKER] Ahmed Ali (98%)        green box + label
    [UNKNOWN]                        red box + label

Pure OpenCV; runs even when face recognition is disabled (identities=[]
is a no-op), so no call site needs feature branches.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# BGR colours.
COLOR_VERIFIED = (80, 220, 80)     # green
COLOR_UNKNOWN = (60, 60, 230)      # red
COLOR_TEXT_BG = (30, 30, 30)
COLOR_TEXT = (255, 255, 255)


def draw_identity_annotations(
    frame: np.ndarray,
    identities: List[Any],
    show_unknown: bool = True,
    show_scores: bool = True,
) -> np.ndarray:
    """
    Draw identity boxes/labels for each FrameIdentity with a bbox.

    Returns a copy; the input frame is never modified.
    """

    annotated = frame.copy()
    height, width = annotated.shape[:2]

    for identity in identities or []:
        bbox = getattr(identity, "bbox", None)
        if not bbox or len(bbox) < 4:
            continue

        try:
            x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
        except (TypeError, ValueError):
            continue

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width - 1))
        y2 = max(y1 + 1, min(y2, height - 1))

        verified = bool(getattr(identity, "verified", False))
        color = COLOR_VERIFIED if verified else COLOR_UNKNOWN

        if not verified and not show_unknown:
            continue

        label = getattr(identity, "label", "Unknown")
        if show_scores and verified:
            score = float(getattr(identity, "score", 0.0) or 0.0)
            label = f"{label} ({score * 100:.0f}%)"

        if not verified:
            label = f"UNKNOWN {label}".strip() if label != "Unknown" else "UNKNOWN"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        text_y = y1 - 6 if y1 - text_h - 6 > 0 else y2 + text_h + 6
        bg_top = text_y - text_h - 4
        cv2.rectangle(
            annotated,
            (x1, max(bg_top, 0)),
            (x1 + text_w + 4, text_y + baseline // 2),
            COLOR_TEXT_BG,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 2, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return annotated


def draw_face_recognition_banner(
    frame: np.ndarray,
    identities: List[Any],
    recognition_enabled: bool,
    workers_registered: int,
    y_offset: int = 42,
) -> np.ndarray:
    """
    Small status banner under the risk banner showing the recognition
    state + this frame's verified identities, e.g.:

        FACES: Ahmed Ali, Unknown
    """

    if not recognition_enabled:
        return frame

    names = []
    for identity in identities or []:
        if identity.verified:
            names.append(identity.worker_name or identity.worker_id or "Worker")
        else:
            names.append("Unknown")

    text = f"FACES [{workers_registered} registered]: "
    text += ", ".join(names) if names else "-"

    cv2.putText(
        frame,
        text,
        (12, y_offset + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 220, 255),
        1,
        cv2.LINE_AA,
    )
    return frame
