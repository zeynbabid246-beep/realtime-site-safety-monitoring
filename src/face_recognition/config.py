"""
Configuration for the face recognition / worker verification layer.

Everything is a dataclass field with an env-var override so nothing is
hardcoded in source. The env prefix is FACE_ (see .env.example). The
.env file in the project root is loaded automatically when python-dotenv
is available (app.settings already does this; loading it here too keeps
this module usable standalone from scripts).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:  # optional at import time; the app already loads .env via app.settings
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional at runtime
    pass

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Relative to the project root. Each worker gets one subfolder:
#   data/face_data/input_images/<worker_id>/*.jpg
FACE_DATA_DIR_DEFAULT = BASE_DIR / "data" / "face_data"
# Default model location: the trained Siamese model from the notebook.
SIAMESE_MODEL_DEFAULT = BASE_DIR / "notebooks" / "siamesemodelv2.h5"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass
class FaceRecognitionConfig:
    """All tunables for the worker-verification layer. Env-overridable."""

    # --- model -------------------------------------------------------
    # Path to the trained Siamese .h5 model (notebook artifact).
    model_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("FACE_MODEL_PATH", str(SIAMESE_MODEL_DEFAULT))
        )
    )
    # Input size the model was trained on (see notebook preprocess()).
    input_size: int = field(default_factory=lambda: _env_int("FACE_INPUT_SIZE", 100))

    # --- worker data --------------------------------------------------
    # Root of the per-worker reference data tree:
    #   <data_dir>/input_images/<worker_id>/<image>.jpg
    #   <data_dir>/workers.json        (worker metadata / registry file)
    #   <data_dir>/embeddings/<worker_id>.npy  (optional embedding cache)
    #   <data_dir>/calibration.json    (calibrated thresholds, if any)
    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("FACE_DATA_DIR", str(FACE_DATA_DIR_DEFAULT)))
    )

    # --- verification thresholds ---------------------------------------
    # Per-pair score above which a comparison counts as a positive match.
    # Default calibrated on this model's own training data (see
    # docs/FACE_RECOGNITION.md): impostor pairs score ~0.0, genuine webcam
    # pairs median ~0.62 -> 0.5 sits squarely in the separation gap. The
    # notebook's 0.9 was far too strict for genuine pairs. Re-calibrate on
    # YOUR workers with scripts/calibrate_threshold.py.
    detection_threshold: float = field(
        default_factory=lambda: _env_float("FACE_DETECTION_THRESHOLD", 0.50)
    )
    # Fraction of a worker's reference images that must score above
    # detection_threshold to accept the identity. With N references the
    # effective rule is ceil(N * 0.6): e.g. 3 refs need all 3, 5 refs need
    # 3. Register 5-10 reference images per worker for robustness.
    verification_threshold: float = field(
        default_factory=lambda: _env_float("FACE_VERIFICATION_THRESHOLD", 0.60)
    )
    # Margin the winner must beat the runner-up by, as a fraction of the
    # runner-up's rate (0 disables the runner-up check entirely).
    min_margin: float = field(
        default_factory=lambda: _env_float("FACE_MIN_MARGIN", 0.0)
    )
    # Ignore reference images whose stored pair-score calibration says
    # they are unreliable (score below this against a random impostor).
    min_reference_quality: float = field(
        default_factory=lambda: _env_float("FACE_MIN_REFERENCE_QUALITY", 0.0)
    )

    # --- per-frame behaviour --------------------------------------------
    # Verify at most once per track per N seconds (verification is slow:
    # one Siamese pass per worker-reference pair). 0 disables caching.
    cache_ttl_s: float = field(
        default_factory=lambda: _env_float("FACE_CACHE_TTL_S", 2.0)
    )
    # Minimum face-crop height in px; smaller crops are skipped (the
    # model was trained on 250x250 webcam crops - tiny faces are noise).
    min_face_size_px: int = field(
        default_factory=lambda: _env_int("FACE_MIN_FACE_SIZE_PX", 40)
    )
    # Head-region fraction of the person box used for the face crop
    # (matches the anatomical head band in src/safety/rules.py).
    head_region_fraction: float = field(
        default_factory=lambda: _env_float("FACE_HEAD_REGION_FRACTION", 0.35)
    )
    # Squeeze the crop horizontally by this factor (head is narrower
    # than shoulders; a small inset keeps co-worker shoulders out).
    face_width_fraction: float = field(
        default_factory=lambda: _env_float("FACE_WIDTH_FRACTION", 0.80)
    )
    # Person-detection confidence below which no face crop is attempted.
    min_person_confidence: float = field(
        default_factory=lambda: _env_float("FACE_MIN_PERSON_CONFIDENCE", 0.45)
    )
    # Maximum faces verified per frame (busy scenes are bandwidth-bound).
    max_faces_per_frame: int = field(
        default_factory=lambda: _env_int("FACE_MAX_FACES_PER_FRAME", 5)
    )
    # Run verification on every Nth frame (1 = every frame).
    frame_stride: int = field(
        default_factory=lambda: _env_int("FACE_FRAME_STRIDE", 1)
    )

    # --- backend selection ----------------------------------------------
    # "siamese" = load siamesemodelv2.h5 with TensorFlow.
    # "stub"    = deterministic no-TF backend for tests / CI.
    backend: str = field(
        default_factory=lambda: os.environ.get("FACE_BACKEND", "siamese")
    )
    # GPU memory growth (must be set before TF initialises the GPU).
    gpu_memory_growth: bool = field(
        default_factory=lambda: os.environ.get("FACE_GPU_MEMORY_GROWTH", "true").lower() != "false"
    )

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        self.data_dir = Path(self.data_dir)

    # ---------------------------------------------------------------
    # Derived paths
    # ---------------------------------------------------------------

    @property
    def input_images_dir(self) -> Path:
        return self.data_dir / "input_images"

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "workers.json"

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def calibration_path(self) -> Path:
        return self.data_dir / "calibration.json"

    def worker_image_dir(self, worker_id: str) -> Path:
        return self.input_images_dir / worker_id

    def ensure_dirs(self) -> None:
        self.input_images_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

    def apply_calibration(self, calibration: Optional[dict]) -> None:
        """
        Override the two thresholds from a calibration result (as written
        by scripts/calibrate_threshold.py / calibration.json). Only keys
        that are present are applied.
        """
        if not calibration:
            return
        if "detection_threshold" in calibration:
            self.detection_threshold = float(calibration["detection_threshold"])
        if "verification_threshold" in calibration:
            self.verification_threshold = float(calibration["verification_threshold"])
