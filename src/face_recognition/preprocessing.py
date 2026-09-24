"""
Preprocessing exactly as the Siamese model was trained on.

The notebook's preprocess() is:
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0

tf.image.resize with default method is bilinear resize WITHOUT
antialiasing - OpenCV's cv2.INTER_AREA differs from that, so this module
uses a plain bilinear cv2.resize, which matches the notebook path
closely (and deterministically) enough that verification behaviour is
unchanged, while keeping preprocessing available without TensorFlow.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

# (height, width) the model was trained on.
INPUT_SIZE = (100, 100)


def crop_with_margin(
    frame: np.ndarray,
    bbox,
    width_fraction: float = 1.0,
    min_size: int = 0,
) -> Optional[np.ndarray]:
    """
    Crop bbox = [x1, y1, x2, y2] out of a BGR frame, clamped to the frame,
    optionally narrowing horizontally to `width_fraction` of the box
    (centred) to keep neighbouring shoulders out of a face crop.

    Returns None when the (clamped) crop would be smaller than min_size
    in either dimension or the bbox is degenerate.
    """

    if frame is None or getattr(frame, "ndim", 0) != 3 or bbox is None:
        return None

    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except (TypeError, ValueError):
        return None

    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return None

    x1 = min(max(x1, 0.0), float(width - 1))
    x2 = min(max(x2, float(x1) + 1.0), float(width))
    y1 = min(max(y1, 0.0), float(height - 1))
    y2 = min(max(y2, float(y1) + 1.0), float(height))

    if width_fraction < 1.0:
        box_w = x2 - x1
        inset = box_w * (1.0 - width_fraction) / 2.0
        x1 += inset
        x2 -= inset

    left, top = int(round(x1)), int(round(y1))
    right, bottom = int(round(x2)), int(round(y2))
    right = max(right, left + 1)
    bottom = max(bottom, top + 1)

    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None
    if min_size > 0 and (crop.shape[0] < min_size or crop.shape[1] < min_size):
        return None
    return crop


def preprocess_face(
    crop: np.ndarray,
    input_size: Tuple[int, int] = INPUT_SIZE,
) -> Optional[np.ndarray]:
    """
    Turn a BGR face crop into the model's expected input:

        BGR -> RGB, resize to input_size bilinearly, scale to [0, 1],
        shape (1, H, W, 3) float32.

    Returns None for empty/invalid crops (never raises for bad input).
    """

    if crop is None or not isinstance(crop, np.ndarray) or crop.ndim != 3 or crop.size == 0:
        return None

    if crop.shape[0] <= 0 or crop.shape[1] <= 0:
        return None

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (input_size[1], input_size[0]), interpolation=cv2.INTER_LINEAR)
    scaled = resized.astype(np.float32) / 255.0
    return np.expand_dims(scaled, axis=0)


def batch_preprocess(
    crops,
    input_size: Tuple[int, int] = INPUT_SIZE,
):
    """
    Preprocess many crops into one batch tensor, skipping invalid crops.

    Returns (batch, valid_indices): batch has shape (N_valid, H, W, 3)
    (or None when no crop is valid); valid_indices maps each batch row
    back to its position in the input list.
    """

    batch_rows = []
    valid_indices = []
    for index, crop in enumerate(crops):
        tensor = preprocess_face(crop, input_size)
        if tensor is None:
            continue
        batch_rows.append(tensor[0])
        valid_indices.append(index)

    if not batch_rows:
        return None, []
    return np.stack(batch_rows, axis=0), valid_indices
