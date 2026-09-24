"""
Face-embedding backends.

SiameseFaceBackend
    Loads the trained notebook model (siamesemodelv2.h5) with
    TensorFlow/Keras. TensorFlow is imported LAZILY inside __init__ so
    that the rest of the package (and the whole safety app) still imports
    on machines without TF.

EmbeddingBackend protocol
    verify_batch(live_batch, reference_batch) -> per-pair scores (N,)
    embed_batch(images)                       -> (N, 4096) embeddings

    The Siamese model cannot be factorised into "embed then compare"
    (the classifier sits on top of the L1 distance of the two 4096-d
    embeddings), so the canonical path is verify_batch on raw score.

StubFaceBackend
    Deterministic, no-TF backend used by unit tests and CI. Its scores
    are driven by simple image statistics so tests can construct
    "same person" / "different person" pairs without weights.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from src.face_recognition.preprocessing import INPUT_SIZE

logger = logging.getLogger(__name__)


def _configure_gpu(memory_growth: bool) -> None:
    """Mirror the notebook's GPU memory-growth setup (best effort)."""

    if not memory_growth:
        return
    try:
        import tensorflow as tf

        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("GPU memory-growth setup skipped: %s", exc)


class SiameseFaceBackend:
    """
    The trained Siamese verification model.

    verify_batch() scores (live, reference) pairs through the full
    network (embedding -> L1 -> sigmoid), exactly like the notebook's
    verify() loop but batched.

    embed_batch() exposes the shared 4096-d embedding branch so a
    registry can cache reference embeddings; scores can then be
    re-derived cheaply (sigmoid(dense(|e_live - e_ref|))) without
    re-running the convolutions.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        input_size: int = INPUT_SIZE[0],
        gpu_memory_growth: bool = True,
    ):
        import tensorflow as tf  # lazy: keeps TF optional at import time
        from tensorflow.keras.layers import Layer

        # Must match the custom layer saved inside the .h5.
        class L1Dist(Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def call(self, input_embedding, validation_embedding):
                return tf.math.abs(input_embedding - validation_embedding)

        self._tf = tf
        _configure_gpu(gpu_memory_growth)

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Siamese face model not found: {model_path}")

        logger.info("Loading Siamese face model: %s", model_path)
        # The .h5 was saved without optimizer state (inference-only), so
        # Keras warns about the missing training config; harmless here.
        self.model = tf.keras.models.load_model(
            str(model_path),
            custom_objects={
                "L1Dist": L1Dist,
                "BinaryCrossentropy": tf.losses.BinaryCrossentropy,
            },
            compile=False,
        )
        self.input_size = int(input_size)

        # The shared embedding branch lives inside the Siamese graph.
        self.embedding_model = self.model.get_layer("embedding")
        # Final Dense(1, sigmoid) after the L1 distance layer.
        self._classifier = self.model.layers[-1]
        self._classifier_weights = self._classifier.get_weights()

        # TF model calls are not thread-safe; the app shares one backend
        # across requests/streams, so serialise forward passes (same
        # pattern as HazardDetector).
        self._lock = threading.RLock()

        logger.info("Siamese face model loaded (input %dx%dx3).", self.input_size, self.input_size)

    # ------------------------------------------------------------------
    # scoring
    # ------------------------------------------------------------------

    def verify_batch(
        self,
        live_batch: np.ndarray,
        reference_batch: np.ndarray,
    ) -> np.ndarray:
        """
        Score N (live, reference) image pairs.

        live_batch / reference_batch: (N, H, W, 3) float32 in [0, 1].
        Returns (N,) float32 scores - higher = more likely same person.
        """

        live_batch = np.asarray(live_batch, dtype=np.float32)
        reference_batch = np.asarray(reference_batch, dtype=np.float32)

        if live_batch.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)

        with self._lock:
            scores = self.model.predict(
                [live_batch, reference_batch], verbose=0
            )
        return np.asarray(scores, dtype=np.float32).reshape(-1)

    def embed_batch(self, images: np.ndarray) -> np.ndarray:
        """
        Compute the shared 4096-d embedding for N preprocessed images.
        images: (N, H, W, 3) float32 in [0, 1]. Returns (N, 4096).
        """

        images = np.asarray(images, dtype=np.float32)
        if images.shape[0] == 0:
            return np.zeros((0, self.embedding_model.output_shape[-1]), dtype=np.float32)

        with self._lock:
            embeddings = self.embedding_model.predict(images, verbose=0)
        return np.asarray(embeddings, dtype=np.float32)

    def score_from_embeddings(self, live_embedding: np.ndarray, reference_embeddings: np.ndarray) -> np.ndarray:
        """
        Re-derive pair scores from cached embeddings without re-running
        the convolutional trunk: sigmoid(Dense(|e_live - e_ref|)).
        """

        live = np.asarray(live_embedding, dtype=np.float32).reshape(1, -1)
        refs = np.asarray(reference_embeddings, dtype=np.float32)
        if refs.size == 0:
            return np.zeros((0,), dtype=np.float32)

        distances = np.abs(live - refs).astype(np.float32)
        weights, biases = self._classifier_weights
        logits = distances @ weights + biases
        return (1.0 / (1.0 + np.exp(-logits))).reshape(-1)

    @property
    def embedding_dim(self) -> int:
        return int(self.embedding_model.output_shape[-1])


class StubFaceBackend:
    """
    Deterministic offline stand-in for SiameseFaceBackend.

    "Embeds" an image as a few downsampled intensity statistics and
    scores pairs by embedding similarity through a sigmoid. Tests use it
    to exercise the whole registry/service/pipeline flow with no TF and
    no trained weights: identical images score ~1, very different images
    score low.
    """

    def __init__(self, input_size: int = INPUT_SIZE[0], embedding_dim: int = 64):
        self.input_size = int(input_size)
        self._embedding_dim = int(embedding_dim)

    def embed_batch(self, images) -> np.ndarray:
        images = np.asarray(images, dtype=np.float32)
        if images.ndim == 3:  # single image
            images = images[np.newaxis, ...]
        if images.shape[0] == 0:
            return np.zeros((0, self._embedding_dim), dtype=np.float32)

        import cv2

        embeddings = []
        grid = int(round(self._embedding_dim ** 0.5))  # 64 dims -> 8x8 grid
        for image in images:
            # (H, W, 3) in [0,1] -> coarse grayscale grid statistics.
            gray = image[..., :3].mean(axis=2)
            small = cv2.resize(gray, (grid * 8, grid * 8), interpolation=cv2.INTER_AREA)
            blocks = small.reshape(grid, 8, grid, 8).mean(axis=(1, 3))
            embeddings.append(blocks.reshape(-1)[: self._embedding_dim])

        result = np.asarray(embeddings, dtype=np.float32)
        if result.shape[1] < self._embedding_dim:
            result = np.pad(result, ((0, 0), (0, self._embedding_dim - result.shape[1])))
        return result

    def verify_batch(self, live_batch, reference_batch) -> np.ndarray:
        """Row-wise pair scores: live[i] vs reference[i]."""

        live_embeddings = self.embed_batch(live_batch)
        reference_embeddings = self.embed_batch(reference_batch)
        return self.score_from_embeddings(live_embeddings, reference_embeddings)

    def score_from_embeddings(self, live_embedding, reference_embeddings) -> np.ndarray:
        """
        Score each reference row against the live embedding(s).

        Accepts live as (D,), (1, D) (broadcast to every reference) or
        (N, D) (row-wise pairs); reference_embeddings is (N, D). Returns
        one score per reference row.
        """

        refs = np.asarray(reference_embeddings, dtype=np.float32)
        if refs.size == 0 or refs.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)

        live = np.asarray(live_embedding, dtype=np.float32)
        if live.ndim == 1:
            live = live.reshape(1, -1)
        if live.shape[0] == 1:
            live = np.repeat(live, refs.shape[0], axis=0)
        if live.shape != refs.shape:
            raise ValueError(
                f"live {live.shape} vs refs {refs.shape}: expected (1,D) or (N,D)"
            )

        distances = np.abs(live - refs).mean(axis=1)
        # distance ~0 (same image) -> ~1.0; distance > ~0.05 -> ~0.0.
        # (np.clip avoids exp overflow in the saturated regime.)
        exponent = np.clip(150.0 * (distances - 0.02), -30.0, 30.0)
        return (1.0 / (1.0 + np.exp(exponent))).astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


def create_backend(config) -> object:
    """
    Build the configured backend.

    "siamese" loads the real .h5 model; anything else (or a missing
    model file with allow_stub=True semantics handled by the caller)
    falls back to StubFaceBackend so tests/CI never need TensorFlow.
    """

    backend_name = str(getattr(config, "backend", "siamese")).lower()
    if backend_name == "stub":
        return StubFaceBackend(input_size=config.input_size)

    if backend_name == "siamese":
        return SiameseFaceBackend(
            config.model_path,
            input_size=config.input_size,
            gpu_memory_growth=config.gpu_memory_growth,
        )

    raise ValueError(f"Unknown face-recognition backend: {backend_name!r}")
