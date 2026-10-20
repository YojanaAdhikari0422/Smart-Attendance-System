"""Transient camera-frame decoding and face descriptor helpers."""

import numpy as np
import cv2


MAX_IMAGE_BYTES = 450_000
MAX_IMAGE_WIDTH = 1280
MAX_IMAGE_HEIGHT = 720
FALLBACK_MATCH_THRESHOLD = 6.0
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if FACE_CASCADE.empty():
    raise RuntimeError("OpenCV could not load its frontal-face cascade.")


class ImageInputError(ValueError):
    """Raised when a transient camera image is invalid or unsuitable."""


def decode_frame(image_bytes):
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageInputError("Camera frame is empty or too large. Use a smaller camera preview.")
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise ImageInputError("Could not read the camera frame. Please try again.")
    height, width = frame.shape[:2]
    if width > MAX_IMAGE_WIDTH or height > MAX_IMAGE_HEIGHT:
        raise ImageInputError("Camera frame dimensions are too large. Please lower the camera preview size.")
    return frame


def get_single_face_embedding(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60),
    )
    if len(faces) == 0:
        raise ImageInputError("No face detected. Center your face in the camera and try again.")
    if len(faces) > 1:
        raise ImageInputError("More than one face detected. Make sure only one person is in the frame.")

    x, y, width, height = (int(value) for value in faces[0])
    crop = frame[max(y, 0):y + height, max(x, 0):x + width]
    if crop.size == 0:
        raise ImageInputError("Could not read the detected face. Please try again.")

    resized = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (32, 32))
    return resized.astype(np.float64).flatten() / 255.0


def best_match(query_embedding, people):
    closest = None
    closest_distance = None
    for person in people:
        known_embedding = np.frombuffer(person.embedding, dtype=np.float64)
        if known_embedding.shape != query_embedding.shape:
            continue
        distance = float(np.linalg.norm(query_embedding - known_embedding))
        if closest_distance is None or distance < closest_distance:
            closest = person
            closest_distance = distance
    if closest is not None and closest_distance < FALLBACK_MATCH_THRESHOLD:
        return closest, closest_distance
    return None, closest_distance


def descriptor_backend():
    return "pixel_fallback"
