"""
Face recognition (worker identity verification) package.

Built around the trained Siamese verification model
(`notebooks/siamesemodelv2.h5` from
`notebooks/Facial Verification with a Siamese Network.ipynb`):

    input_img, validation_img (100x100x3, [0,1])
        -> shared 'embedding' model (4096-d sigmoid)
        -> L1 distance layer
        -> Dense(1, sigmoid) -> verification score in [0, 1]

The score is the probability that the two crops show the SAME person.
A face is verified as worker W when enough comparisons against W's
reference images score above FACE_DETECTION_THRESHOLD.

The package is model-free at import time: the TensorFlow backend is only
imported when a SiameseFaceBackend is instantiated, and every pure-Python
component (registry, crops, matching, calibration, overlay) is unit
testable without TF.

Modules
-------
config          FaceRecognitionConfig - every tunable, env-overridable.
preprocessing   Notebook-exact crop resizing/normalisation.
backends        SiameseFaceBackend (TF) + StubFaceBackend (tests).
registry        WorkerRegistry - workers, reference images, embeddings.
face_detector   FaceDetector - YOLO person boxes -> face crops.
service         FaceRecognitionService - verify faces against all workers.
overlay         draw_identity_annotations - HUD for verified workers.
calibration     calibrate_threshold - score-driven threshold calibration.
"""

from src.face_recognition.config import FaceRecognitionConfig
from src.face_recognition.registry import WorkerRegistry, WorkerRecord

__all__ = ["FaceRecognitionConfig", "WorkerRegistry", "WorkerRecord"]
