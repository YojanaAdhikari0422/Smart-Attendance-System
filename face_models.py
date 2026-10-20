"""OpenCV YuNet/SFace model setup and CPU face recognition helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import threading
from urllib.request import Request, urlopen

import cv2
import numpy as np


MODEL_DIRECTORY = Path(__file__).resolve().with_name("models")
MODEL_FILES = {
    "face_detection_yunet_2023mar.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
}
SFACE_EMBEDDING_SIZE = 128
SFACE_COSINE_THRESHOLD = 0.363


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_files(model_directory: Path = MODEL_DIRECTORY) -> list[Path]:
    """Ensure both pinned OpenCV Zoo models exist and match their SHA-256 digests."""
    model_directory.mkdir(parents=True, exist_ok=True)
    verified = []
    for filename, (url, expected_digest) in MODEL_FILES.items():
        destination = model_directory / filename
        if destination.is_file() and _sha256(destination) == expected_digest:
            verified.append(destination)
            continue

        temporary_path = None
        digest = hashlib.sha256()
        try:
            request = Request(url, headers={"User-Agent": "SmartAttendance/1.0"})
            with urlopen(request, timeout=60) as response:
                with tempfile.NamedTemporaryFile(
                    prefix=f".{destination.stem}-",
                    suffix=".download",
                    dir=model_directory,
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    while chunk := response.read(1024 * 1024):
                        digest.update(chunk)
                        temporary_file.write(chunk)

            if digest.hexdigest() != expected_digest:
                raise RuntimeError(
                    f"SHA-256 verification failed for {filename}; "
                    "the downloaded file was discarded."
                )

            os.replace(temporary_path, destination)
            verified.append(destination)
        except (OSError, RuntimeError, ValueError):
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
    return verified


class OpenCVFaceRecognizer:
    """CPU-based YuNet detector and SFace multi-view embedding encoder."""

    def __init__(self, model_directory: Path = MODEL_DIRECTORY):
        detector_path = model_directory / "face_detection_yunet_2023mar.onnx"
        recognizer_path = model_directory / "face_recognition_sface_2021dec.onnx"
        if not detector_path.is_file() or not recognizer_path.is_file():
            raise FileNotFoundError(
                "OpenCV face-model files are missing. Run `python download_face_models.py`."
            )

        if not hasattr(cv2, "FaceDetectorYN_create") or not hasattr(cv2, "FaceRecognizerSF_create"):
            raise RuntimeError(
                "This OpenCV installation lacks the YuNet/SFace APIs. "
                "Install opencv-contrib-python-headless>=4.10,<5 in the desktop environment."
            )

        self.detector = cv2.FaceDetectorYN_create(
            str(detector_path),
            "",
            (320, 320),
            0.75,
            0.3,
            5000,
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )
        self.recognizer = cv2.FaceRecognizerSF_create(
            str(recognizer_path),
            "",
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )

    def detect(self, frame: np.ndarray) -> list[np.ndarray]:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return []
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
        return [face for face in faces if face[2] >= 40 and face[3] >= 40]

    def encode(self, frame: np.ndarray, face: np.ndarray) -> np.ndarray:
        aligned_face = self.recognizer.alignCrop(frame, face)
        feature = self.recognizer.feature(aligned_face).reshape(-1).astype(np.float32)
        if feature.shape != (SFACE_EMBEDDING_SIZE,):
            raise ValueError("SFace returned an unexpected face descriptor.")
        norm = float(np.linalg.norm(feature))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("SFace could not create a valid face descriptor.")
        return feature / norm

    def encode_enrollment_photo(self, image_path: str | Path) -> np.ndarray:
        image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {Path(image_path).name}")
        faces = self.detect(image)
        if len(faces) != 1:
            raise ValueError(
                f"{Path(image_path).name}: expected one visible face, found {len(faces)}."
            )
        return self.encode(image, faces[0])


_recognizer = None
_recognizer_lock = threading.Lock()


def get_face_recognizer() -> OpenCVFaceRecognizer:
    global _recognizer
    if _recognizer is None:
        with _recognizer_lock:
            if _recognizer is None:
                _recognizer = OpenCVFaceRecognizer()
    return _recognizer
