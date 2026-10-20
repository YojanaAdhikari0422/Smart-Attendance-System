"""
Smart Attendance System - Desktop Application
================================================
PyQt5 + OpenCV + SQLite face-recognition attendance system.

Architecture
------------
- LoginDialog            : Admin gate configured through local environment variables.
- MainWindow             : Branded desktop shell -> sidebar navigation + QStackedWidget pages.
- CameraThread (QThread) : Runs the live camera + recognition loop off the UI thread.
- Database               : Thin SQLite wrapper (users + attendance tables).
- LiveRecognitionWidget  : Module 1 - start/stop camera, live overlay, auto attendance marking.
                           Gated by an OpenCV-driven liveness check (blink detection via
                           MediaPipe FaceMesh landmarks + Eye Aspect Ratio) - a face must
                           blink at least once before it can be matched and marked present.
- RegistrationWidget     : Module 2 - capture faces, compute mean embedding, full user CRUD.
- AnalyticsWidget        : Module 3 - attendance log table + per-student presence chart.
- ExportWidget           : Module 4 - export attendance to .xlsx filtered by week/month/year.

Run with:
    pip install PyQt5 opencv-python numpy pandas openpyxl mediapipe
    # optional, for real face recognition instead of the built-in fallback:
    pip install face_recognition
    python run_attendance.py

Note: MediaPipe (not dlib) powers the liveness landmarks, since dlib requires compiling
C++ on Windows. If mediapipe isn't installed, the app still runs - it just skips the
liveness gate and matches faces directly (a status message says so on camera start).
"""

import sys
import os
import sqlite3
import threading
import tempfile
from datetime import datetime, date, timedelta
from pathlib import Path

from dotenv import load_dotenv

import numpy as np
import cv2

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QDialog, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QFormLayout, QListWidget, QListWidgetItem,
    QStackedWidget, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QComboBox, QFileDialog, QFrame, QAbstractItemView
)

# ======================================================================================
# CONFIG / CONSTANTS
# ======================================================================================

load_dotenv(Path(__file__).resolve().with_name(".env"))

DB_PATH = str(Path(__file__).resolve().with_name("attendance_system.db"))
ATTENDANCE_RECORDS_DIR = Path(__file__).resolve().with_name("Attendance_records")
ADMIN_USERNAME = os.environ.get("ATTENDANCE_ADMIN_USERNAME", "")
ADMIN_PASSWORD = os.environ.get("ATTENDANCE_ADMIN_PASSWORD", "")
CAPTURE_COUNT = 6          # number of frames captured per registration
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

APP_STYLESHEET = """
QWidget {
    color: #172033;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QMainWindow, QDialog {
    background: #f3f5f9;
}
QLabel#mutedLabel {
    color: #6c7890;
}
QLabel#pageTitle {
    color: #172033;
    font-size: 20pt;
    font-weight: 700;
}
QLabel#pageSubtitle {
    color: #71809a;
    font-size: 10pt;
}
QPushButton {
    background: #ffffff;
    color: #25334d;
    border: 1px solid #d9e0eb;
    border-radius: 7px;
    padding: 8px 14px;
    min-height: 20px;
    font-weight: 600;
}
QPushButton:hover {
    background: #f0f5ff;
    border-color: #91b5ff;
}
QPushButton:pressed {
    background: #e5edfb;
}
QPushButton:disabled {
    color: #a3adbd;
    background: #edf0f5;
    border-color: #e2e6ed;
}
QPushButton#primaryButton {
    background: #315fce;
    color: #ffffff;
    border-color: #315fce;
}
QPushButton#primaryButton:hover {
    background: #264fae;
    border-color: #264fae;
}
QPushButton#dangerButton {
    color: #b42332;
}
QLineEdit, QComboBox {
    background: #ffffff;
    border: 1px solid #d9e0eb;
    border-radius: 7px;
    padding: 8px 10px;
    min-height: 20px;
    selection-background-color: #315fce;
}
QLineEdit:focus, QComboBox:focus {
    border: 1px solid #5785e6;
}
QGroupBox {
    background: #ffffff;
    border: 1px solid #e1e6ee;
    border-radius: 10px;
    margin-top: 12px;
    padding: 16px 12px 12px 12px;
    font-weight: 700;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
    color: #33435e;
}
QTableWidget {
    background: #ffffff;
    alternate-background-color: #f7f9fc;
    border: 1px solid #e4e9f0;
    border-radius: 7px;
    gridline-color: #edf0f5;
    selection-background-color: #e8efff;
    selection-color: #172033;
}
QHeaderView::section {
    background: #f5f7fa;
    color: #64728a;
    border: none;
    border-bottom: 1px solid #e4e9f0;
    padding: 9px 8px;
    font-weight: 700;
}
QListWidget#navigation {
    background: #17243a;
    color: #c2cde0;
    border: none;
    outline: none;
    padding: 10px 8px;
}
QListWidget#navigation::item {
    border-radius: 7px;
    padding: 12px 12px;
    margin: 2px 4px;
}
QListWidget#navigation::item:hover {
    background: #223451;
    color: #ffffff;
}
QListWidget#navigation::item:selected {
    background: #315fce;
    color: #ffffff;
}
QFrame#topBar, QFrame#metricCard, QFrame#contentCard {
    background: #ffffff;
    border: 1px solid #e1e6ee;
    border-radius: 10px;
}
QFrame#sidebar {
    background: #17243a;
    border: none;
}
QLabel#videoPreview {
    background: #101827;
    color: #c6d0e0;
    border: 1px solid #27344a;
    border-radius: 10px;
}
"""


def style_table(table):
    table.setAlternatingRowColors(True)
    table.setShowGrid(False)
    table.setWordWrap(False)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.verticalHeader().setVisible(False)
    table.verticalHeader().setDefaultSectionSize(38)
    table.horizontalHeader().setMinimumHeight(38)


# ======================================================================================
# OPTIONAL FACE RECOGNITION IMPORT (with a working fallback so the UI always runs)
# ======================================================================================

try:
    import face_recognition
    HAS_FACE_RECOGNITION = True
    MATCH_THRESHOLD = 0.6          # good default for face_recognition's 128-d encodings
except (ImportError, RuntimeError):
    face_recognition = None
    HAS_FACE_RECOGNITION = False
    MATCH_THRESHOLD = 6.0          # tuned for the 32x32 grayscale fallback below

FACE_CASCADE = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# ======================================================================================
# LIVENESS DETECTION (OpenCV-driven blink check via MediaPipe FaceMesh landmarks)
# ======================================================================================
# MediaPipe is used instead of dlib for landmarks - dlib requires compiling C++ on
# Windows (CMake/Visual Studio), which is exactly the install pain you hit earlier.
# MediaPipe installs with a plain `pip install mediapipe`, no compiler needed.
#
# The liveness signal itself is a classic OpenCV-style Eye Aspect Ratio (EAR) blink
# check: a face is only trusted enough to mark attendance once at least one blink has
# been observed. This stops a printed photo held up to the camera; it will NOT stop a
# video replay of the same person blinking - that would need a trained anti-spoofing
# model (see the separate DeepFace-based notebook if you need that stronger guarantee).

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
    _mp_face_mesh_model = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=5,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    # 6-point eye landmark indices (outer corner, top1, top2, inner corner, bottom1, bottom2)
    RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
    LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
except ImportError:
    HAS_MEDIAPIPE = False
    _mp_face_mesh_model = None

EAR_OPEN_MIN = 0.25       # eyes must clearly read as OPEN before/after a blink
EAR_CLOSED_MAX = 0.18     # eyes must clearly read as CLOSED during a blink
                          # (gap between the two = hysteresis, so noise sitting near a
                          # single threshold can't flip state back and forth repeatedly)
MIN_OPEN_FRAMES = 3       # sustained frames needed to confirm "reopened" after closing
MIN_CLOSED_FRAMES = 2     # sustained frames needed to confirm a real closure
REQUIRED_BLINKS = 2       # two full open->closed->open cycles needed before trusting a face
EAR_SMOOTHING_WINDOW = 3  # rolling average size to damp single-frame landmark jitter


def eye_aspect_ratio(eye_points):
    """Standard EAR formula: vertical eye distances over horizontal eye distance."""
    a = np.linalg.norm(eye_points[1] - eye_points[5])
    b = np.linalg.norm(eye_points[2] - eye_points[4])
    c = np.linalg.norm(eye_points[0] - eye_points[3])
    return (a + b) / (2.0 * c) if c > 0 else 0.0


def get_eye_landmarks(frame_bgr):
    """
    Runs MediaPipe FaceMesh on the frame and returns a list of (center_x, center_y, ear)
    tuples, one per detected face mesh. Returns [] if MediaPipe isn't installed or no
    face mesh is found, so callers can gracefully skip the liveness gate.
    """
    if not HAS_MEDIAPIPE:
        return []

    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = _mp_face_mesh_model.process(rgb)
    if not results.multi_face_landmarks:
        return []

    output = []
    for face_landmarks in results.multi_face_landmarks:
        pts = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])
        left_eye = pts[LEFT_EYE_IDX]
        right_eye = pts[RIGHT_EYE_IDX]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        output.append((cx, cy, ear))
    return output


# ======================================================================================
# FACE UTILITIES
# ======================================================================================

def detect_faces(frame_bgr):
    """Return a list of (x, y, w, h) boxes using a Haar cascade (works regardless of
    which encoding backend is active, so both branches share the same detection step)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    return faces


def get_face_encoding(frame_bgr, box):
    """
    Compute a face embedding for the given (x, y, w, h) box.
    Uses face_recognition's 128-d encoding when available; otherwise falls back to a
    deterministic 1024-d normalized-pixel descriptor so the whole pipeline still works
    end-to-end without any extra dependency.
    """
    x, y, w, h = box
    face_crop = frame_bgr[max(y, 0):y + h, max(x, 0):x + w]
    if face_crop.size == 0:
        return None

    if HAS_FACE_RECOGNITION:
        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_crop)
        if encodings:
            return encodings[0].astype(np.float64)
        return None
    else:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32)).astype(np.float64) / 255.0
        return resized.flatten()


def find_best_match(embedding, known_embeddings, threshold):
    """known_embeddings: dict student_id -> (name, embedding ndarray). Returns (student_id, name, distance) or (None, None, None)."""
    if embedding is None or not known_embeddings:
        return None, None, None

    best_id, best_name, best_dist = None, None, None
    for sid, (name, known_emb) in known_embeddings.items():
        if known_emb is None or known_emb.shape != embedding.shape:
            continue
        d = float(np.linalg.norm(embedding - known_emb))
        if best_dist is None or d < best_dist:
            best_id, best_name, best_dist = sid, name, d

    if best_dist is not None and best_dist < threshold:
        return best_id, best_name, best_dist
    return None, None, None


def cv_to_qpixmap(frame_bgr):
    """Convert an OpenCV BGR frame into a QPixmap for display in a QLabel."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ======================================================================================
# DATABASE LAYER
# ======================================================================================

class Database:
    """SQLite wrapper shared between the UI thread and the camera QThread."""

    def __init__(self, path=DB_PATH):
        self.path = path
        # check_same_thread=False because the camera thread also writes attendance.
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    student_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    class_name TEXT NOT NULL DEFAULT 'Unassigned'
                )
            """)
            columns = {row[1] for row in cur.execute("PRAGMA table_info(users)")}
            if "class_name" not in columns:
                cur.execute(
                    "ALTER TABLE users ADD COLUMN class_name TEXT NOT NULL DEFAULT 'Unassigned'"
                )
            cur.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    date DATE NOT NULL,
                    first_seen TIME NOT NULL,
                    last_seen TIME NOT NULL,
                    FOREIGN KEY(student_id) REFERENCES users(student_id)
                )
            """)
            self.conn.commit()

    # ---------------------------- USERS (CRUD) ----------------------------

    def add_user(self, student_id, name, class_name, embedding):
        blob = np.asarray(embedding, dtype=np.float64).tobytes()
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO users (student_id, name, class_name, embedding) VALUES (?, ?, ?, ?)",
                (student_id, name, class_name, blob)
            )
            self.conn.commit()

    def update_user(self, student_id, name=None, class_name=None, embedding=None):
        with self.lock:
            cur = self.conn.cursor()
            if name is not None:
                cur.execute("UPDATE users SET name=? WHERE student_id=?", (name, student_id))
            if class_name is not None:
                cur.execute(
                    "UPDATE users SET class_name=? WHERE student_id=?",
                    (class_name, student_id),
                )
            if embedding is not None:
                blob = np.asarray(embedding, dtype=np.float64).tobytes()
                cur.execute("UPDATE users SET embedding=? WHERE student_id=?", (blob, student_id))
            self.conn.commit()

    def delete_user(self, student_id):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM users WHERE student_id=?", (student_id,))
            cur.execute("DELETE FROM attendance WHERE student_id=?", (student_id,))
            self.conn.commit()

    def get_all_users(self):
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT student_id, name, class_name FROM users ORDER BY name")
            return cur.fetchall()

    def get_all_embeddings(self):
        """Returns dict: student_id -> (name, embedding ndarray). Loaded into RAM by the camera thread."""
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT student_id, name, embedding FROM users")
            rows = cur.fetchall()
        result = {}
        for sid, name, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float64)
            result[sid] = (name, emb)
        return result

    # ---------------------------- ATTENDANCE ----------------------------

    def mark_attendance(self, student_id):
        today_str = date.today().isoformat()
        now_str = datetime.now().strftime("%H:%M:%S")
        with self.lock:
            cur = self.conn.cursor()
            cur.execute(
                "SELECT id FROM attendance WHERE student_id=? AND date=?",
                (student_id, today_str)
            )
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    "INSERT INTO attendance (student_id, date, first_seen, last_seen) VALUES (?, ?, ?, ?)",
                    (student_id, today_str, now_str, now_str)
                )
            else:
                cur.execute(
                    "UPDATE attendance SET last_seen=? WHERE id=?",
                    (now_str, row[0])
                )
            self.conn.commit()
            cur.execute(
                """
                SELECT u.class_name, a.date, a.first_seen
                FROM attendance a
                JOIN users u ON u.student_id = a.student_id
                WHERE a.student_id=? AND a.date=?
                """,
                (student_id, today_str),
            )
            record = cur.fetchone()
            if record is None:
                raise RuntimeError("Attendance was written but its student record could not be loaded.")
            return record

    def get_attendance(self, start_date=None, end_date=None):
        """Returns student ID, name, class, date, first seen, and last seen, newest first."""
        query = """
            SELECT a.student_id, u.name, u.class_name, a.date, a.first_seen, a.last_seen
            FROM attendance a
            LEFT JOIN users u ON u.student_id = a.student_id
        """
        params = []
        if start_date and end_date:
            query += " WHERE a.date BETWEEN ? AND ?"
            params = [start_date, end_date]
        query += " ORDER BY a.date DESC, a.first_seen DESC"

        with self.lock:
            cur = self.conn.cursor()
            cur.execute(query, params)
            return cur.fetchall()

    def get_attendance_percentage(self):
        """
        Per-student presence percentage = (distinct days that student was marked present)
        / (distinct days anyone was marked present in the whole system) * 100.
        """
        with self.lock:
            cur = self.conn.cursor()
            cur.execute("SELECT DISTINCT date FROM attendance")
            all_days = [r[0] for r in cur.fetchall()]
            total_days = len(all_days) if all_days else 1

            cur.execute("SELECT student_id, name, class_name FROM users")
            users = cur.fetchall()

            result = []
            for sid, name, class_name in users:
                cur.execute("SELECT COUNT(DISTINCT date) FROM attendance WHERE student_id=?", (sid,))
                present_days = cur.fetchone()[0]
                pct = round((present_days / total_days) * 100, 1)
                result.append((sid, name, class_name, present_days, total_days, pct))
        return result


def _excel_sheet_name(class_name, existing_names):
    """Return a valid, unique Excel worksheet name for a class."""
    base_name = "".join(
        "_" if character in "[]:*?/\\"
        else character
        for character in (class_name.strip() or "Unassigned")
    ).strip("'") or "Unassigned"
    base_name = base_name[:31]
    existing = {name.casefold() for name in existing_names}
    if base_name.casefold() not in existing:
        return base_name

    suffix_number = 2
    while True:
        suffix = f" ({suffix_number})"
        candidate = f"{base_name[:31 - len(suffix)]}{suffix}"
        if candidate.casefold() not in existing:
            return candidate
        suffix_number += 1


def write_daily_attendance_record(student_id, name, class_name, attendance_date, attendance_time):
    """Create or update one student row in the class sheet for the given date."""
    from openpyxl import Workbook, load_workbook

    day = date.fromisoformat(attendance_date)
    ATTENDANCE_RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    workbook_path = ATTENDANCE_RECORDS_DIR / f"{day.strftime('%d-%m-%Y')}.xlsx"
    class_name = class_name.strip() or "Unassigned"
    headers = ["Name", "Roll Number", "Class", "Attendance Time"]

    if workbook_path.is_file():
        workbook = load_workbook(workbook_path)
    else:
        workbook = Workbook()

    matching_sheet = None
    existing_roll_rows = []
    for worksheet in workbook.worksheets:
        for row_index in range(2, worksheet.max_row + 1):
            if worksheet.cell(row_index, 3).value == class_name and matching_sheet is None:
                matching_sheet = worksheet
            if worksheet.cell(row_index, 2).value == student_id:
                existing_roll_rows.append((worksheet, row_index))

    if matching_sheet is None:
        matching_sheet = next(
            (
                worksheet for worksheet in workbook.worksheets
                if worksheet.max_row == 1
                and worksheet.cell(1, 1).value is None
            ),
            None,
        )
        if matching_sheet is not None:
            existing_names = {
                name for name in workbook.sheetnames if name != matching_sheet.title
            }
            matching_sheet.title = _excel_sheet_name(class_name, existing_names)
        else:
            matching_sheet = workbook.create_sheet(
                _excel_sheet_name(class_name, workbook.sheetnames)
            )

    if tuple(matching_sheet.cell(1, column).value for column in range(1, 5)) != tuple(headers):
        if matching_sheet.max_row != 1 or matching_sheet.cell(1, 1).value is not None:
            raise ValueError(
                f"Worksheet '{matching_sheet.title}' has unexpected columns; "
                "the workbook was left unchanged."
            )
        for column, header in enumerate(headers, start=1):
            matching_sheet.cell(1, column, header)

    matching_roll_rows = [
        row_index for worksheet, row_index in existing_roll_rows
        if worksheet is matching_sheet
    ]
    if matching_roll_rows:
        row_index = matching_roll_rows[0]
        matching_sheet.cell(row_index, 1, name)
        matching_sheet.cell(row_index, 2, student_id)
        matching_sheet.cell(row_index, 3, class_name)
        matching_sheet.cell(row_index, 4, attendance_time)
    else:
        matching_sheet.append([name, student_id, class_name, attendance_time])

    for worksheet, row_index in sorted(
        existing_roll_rows,
        key=lambda row: (id(row[0]), row[1]),
        reverse=True,
    ):
        if worksheet is matching_sheet and row_index == (matching_roll_rows[0] if matching_roll_rows else None):
            continue
        worksheet.delete_rows(row_index)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{workbook_path.stem}-",
            suffix=".xlsx",
            dir=ATTENDANCE_RECORDS_DIR,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        workbook.save(temporary_path)
        os.replace(temporary_path, workbook_path)
    except (OSError, ValueError):
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return workbook_path


# ======================================================================================
# CAMERA THREAD (Module 1 - live recognition, runs off the UI thread)
# ======================================================================================

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    recognized = pyqtSignal(str, str, str, str, str)
    status_changed = pyqtSignal(str)

    def __init__(self, db: Database, parent=None):
        super().__init__(parent)
        self.db = db
        self._running = False
        self.known_embeddings = {}
        self._last_marked = {}     # student_id -> last mark_attendance call time (avoid spamming DB every frame)
        self._blink_trackers = {}  # coarse position key -> {"consec": int, "blinks": int}
        self.liveness_enabled = HAS_MEDIAPIPE

    def refresh_known_faces(self):
        self.known_embeddings = self.db.get_all_embeddings()

    @staticmethod
    def _tracker_key(cx, cy):
        # Coarse spatial bucket so a blink counter persists for the same face across
        # frames without needing a full object tracker - fine for attendance use where
        # people stand still in front of the camera for a moment.
        return (int(cx) // 40, int(cy) // 40)

    def _update_blink_count(self, cx, cy, raw_ear):
        """
        State-machine blink detector: requires a clean OPEN -> CLOSED -> OPEN cycle,
        each phase sustained for several frames, with a hysteresis gap between the
        open/closed thresholds. This is intentionally stricter than a single
        threshold crossing, so ordinary landmark jitter on a static photo (autofocus
        hunting, hand tremor, JPEG noise) can't be mistaken for a real blink.
        """
        key = self._tracker_key(cx, cy)
        tracker = self._blink_trackers.setdefault(key, {
            "state": "OPEN", "open_frames": 0, "closed_frames": 0,
            "blinks": 0, "ear_history": []
        })

        # Smooth the EAR signal with a short rolling average to damp single-frame noise.
        hist = tracker["ear_history"]
        hist.append(raw_ear)
        if len(hist) > EAR_SMOOTHING_WINDOW:
            hist.pop(0)
        ear = sum(hist) / len(hist)

        if tracker["state"] == "OPEN":
            if ear < EAR_CLOSED_MAX:
                tracker["closed_frames"] += 1
                if tracker["closed_frames"] >= MIN_CLOSED_FRAMES:
                    tracker["state"] = "CLOSED"
                    tracker["open_frames"] = 0
            else:
                tracker["closed_frames"] = 0
        elif tracker["state"] == "CLOSED":
            if ear > EAR_OPEN_MIN:
                tracker["open_frames"] += 1
                if tracker["open_frames"] >= MIN_OPEN_FRAMES:
                    tracker["blinks"] += 1
                    tracker["state"] = "OPEN"
                    tracker["closed_frames"] = 0
            else:
                tracker["open_frames"] = 0

        return tracker["blinks"]

    def run(self):
        self._running = True
        self.refresh_known_faces()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_changed.emit("Could not open webcam.")
            self._running = False
            return

        if not HAS_MEDIAPIPE:
            self.status_changed.emit(
                "Camera running (liveness check disabled - install 'mediapipe' to enable blink detection)."
            )
        else:
            self.status_changed.emit("Camera running. Liveness check active (blink required).")

        while self._running:
            ret, frame = cap.read()
            if not ret:
                self.status_changed.emit("Lost camera feed.")
                break

            faces = detect_faces(frame)
            eye_landmarks = get_eye_landmarks(frame)  # [] if mediapipe not installed

            for (x, y, w, h) in faces:
                box_cx, box_cy = x + w / 2, y + h / 2

                # Find the closest MediaPipe face mesh to this Haar box, so we know
                # which blink counter belongs to which detected face.
                is_live = not self.liveness_enabled   # if liveness is unavailable, don't block matching
                blink_count = 0
                if self.liveness_enabled and eye_landmarks:
                    closest = min(eye_landmarks, key=lambda e: (e[0] - box_cx) ** 2 + (e[1] - box_cy) ** 2)
                    ex, ey, ear = closest
                    blink_count = self._update_blink_count(ex, ey, ear)
                    is_live = blink_count >= REQUIRED_BLINKS

                name_label = f"Verifying... (blink twice to confirm, {blink_count}/{REQUIRED_BLINKS})"
                color = (0, 255, 255)  # yellow while waiting for a blink

                if is_live:
                    embedding = get_face_encoding(frame, (x, y, w, h))
                    sid, name, _ = find_best_match(embedding, self.known_embeddings, MATCH_THRESHOLD)

                    if sid is not None:
                        name_label = f"{name} (LIVE)"
                        color = (0, 200, 0)

                        # Throttle DB writes to once every few seconds per person, not every frame.
                        last_time = self._last_marked.get(sid, 0)
                        if time_now() - last_time > 3:
                            class_name, attendance_date, attendance_time = self.db.mark_attendance(sid)
                            self._last_marked[sid] = time_now()
                            self.recognized.emit(
                                sid, name, class_name, attendance_date, attendance_time
                            )
                    else:
                        name_label = "Unknown (LIVE)"
                        color = (0, 0, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, name_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            self.frame_ready.emit(frame)
            self.msleep(30)

        cap.release()
        self.status_changed.emit("Camera stopped.")

    def stop(self):
        self._running = False
        self.wait()


def time_now():
    return datetime.now().timestamp()


# ======================================================================================
# LOGIN DIALOG (Admin Authentication Gate)
# ======================================================================================

class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Admin Login - Smart Attendance System")
        self.setFixedSize(420, 360)
        self.authenticated = False

        brand = QLabel("SMART ATTENDANCE")
        brand.setStyleSheet("color: #315fce; font-size: 10pt; font-weight: 700; letter-spacing: 1px;")
        brand.setAlignment(Qt.AlignCenter)

        title = QLabel("Welcome back")
        title.setObjectName("pageTitle")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Sign in to manage attendance and enrollment.")
        subtitle.setObjectName("pageSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter your username")

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter your password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.returnPressed.connect(self.try_login)
        self.username_input.setFocus()

        login_btn = QPushButton("Sign in")
        login_btn.setObjectName("primaryButton")
        login_btn.setDefault(True)
        login_btn.clicked.connect(self.try_login)

        form = QFormLayout()
        form.setVerticalSpacing(12)
        form.addRow("Username", self.username_input)
        form.addRow("Password", self.password_input)

        layout = QVBoxLayout()
        layout.setContentsMargins(36, 28, 36, 28)
        layout.setSpacing(12)
        layout.addStretch()
        layout.addWidget(brand)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(8)
        layout.addLayout(form)
        layout.addWidget(login_btn)
        layout.addStretch()
        self.setLayout(layout)

    def try_login(self):
        user = self.username_input.text().strip()
        pwd = self.password_input.text().strip()
        if user == ADMIN_USERNAME and pwd == ADMIN_PASSWORD:
            self.authenticated = True
            self.accept()
        else:
            QMessageBox.warning(self, "Login Failed", "Incorrect username or password.")
            self.password_input.clear()


# ======================================================================================
# PAGE 1: LIVE RECOGNITION & ATTENDANCE TRACKING
# ======================================================================================

class LiveRecognitionWidget(QWidget):
    def __init__(self, db: Database):
        super().__init__()
        self.db = db
        self.camera_thread = None
        self._exported_attendance = set()

        self.video_label = QLabel("Camera is stopped.\nStart the camera to begin recognition.")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(520, 360)
        self.video_label.setObjectName("videoPreview")

        self.status_label = QLabel("Status: idle")
        self.status_label.setObjectName("mutedLabel")

        self.last_recognized_label = QLabel("Last recognized: -")
        self.last_recognized_label.setWordWrap(True)

        self.start_btn = QPushButton("Start Camera")
        self.start_btn.setObjectName("primaryButton")
        self.start_btn.clicked.connect(self.start_camera)

        self.stop_btn = QPushButton("Stop Camera")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)

        header = QLabel("Live attendance")
        header.setObjectName("pageTitle")
        subtitle = QLabel("Recognize registered people and record attendance in real time.")
        subtitle.setObjectName("pageSubtitle")

        liveness_note = QLabel(
            "Liveness verification is active. Blink twice to confirm a person before attendance is marked."
            if HAS_MEDIAPIPE else
            "Liveness verification is disabled. Install mediapipe to require blink verification."
        )
        liveness_note.setObjectName("mutedLabel")
        liveness_note.setWordWrap(True)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(3)
        title_layout.addWidget(header)
        title_layout.addWidget(subtitle)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()

        session_box = QGroupBox("Session status")
        session_layout = QVBoxLayout()
        session_layout.setSpacing(12)
        session_layout.addWidget(self.status_label)
        session_layout.addWidget(self.last_recognized_label)
        session_layout.addSpacing(8)
        session_layout.addWidget(liveness_note)
        session_layout.addStretch()
        session_box.setLayout(session_layout)
        session_box.setMinimumWidth(230)

        content = QHBoxLayout()
        content.setSpacing(18)
        content.addWidget(self.video_label, 3)
        content.addWidget(session_box, 1)

        layout = QVBoxLayout()
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)
        layout.addLayout(title_layout)
        layout.addLayout(btn_row)
        layout.addLayout(content, 1)
        self.setLayout(layout)

    def start_camera(self):
        if self.camera_thread is not None and self.camera_thread.isRunning():
            return
        self.camera_thread = CameraThread(self.db)
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.recognized.connect(self.on_recognized)
        self.camera_thread.status_changed.connect(self.on_status)
        self.camera_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_camera(self):
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread = None
        self.video_label.setText("Camera is stopped.")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_frame(self, frame):
        pixmap = cv_to_qpixmap(frame)
        self.video_label.setPixmap(
            pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio)
        )

    def on_recognized(self, student_id, name, class_name, attendance_date, attendance_time):
        self.last_recognized_label.setText(
            f"Last recognized: {name} ({student_id}) · {class_name} · {attendance_time}"
        )
        export_key = (attendance_date, student_id)
        workbook_path = (
            ATTENDANCE_RECORDS_DIR
            / f"{date.fromisoformat(attendance_date).strftime('%d-%m-%Y')}.xlsx"
        )
        if export_key in self._exported_attendance and workbook_path.is_file():
            return

        try:
            write_daily_attendance_record(
                student_id,
                name,
                class_name,
                attendance_date,
                attendance_time,
            )
        except (OSError, ValueError) as error:
            self.status_label.setText(
                f"Attendance was recorded, but its Excel file could not be updated: {error}"
            )
            return

        self._exported_attendance.add(export_key)
        self.status_label.setText(
            f"Attendance saved to {workbook_path.relative_to(Path(__file__).resolve().parent)}"
        )

    def on_status(self, message):
        self.status_label.setText(f"Status: {message}")
        self.start_btn.setEnabled(not message.startswith("Camera running"))
        self.stop_btn.setEnabled(message.startswith("Camera running"))

    def refresh_known_faces(self):
        """Called after registration/CRUD changes so a running camera picks up new users."""
        if self.camera_thread is not None:
            self.camera_thread.refresh_known_faces()
        self._exported_attendance.clear()


# ======================================================================================
# PAGE 2: PERSON REGISTRATION & FACE CRUD MANAGEMENT
# ======================================================================================

class RegistrationWidget(QWidget):
    def __init__(self, db: Database, on_users_changed=None):
        super().__init__()
        self.db = db
        self.on_users_changed = on_users_changed  # callback to notify MainWindow / live widget

        self.capture_cap = None
        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self._update_capture_preview)
        self.captured_embeddings = []
        self.editing_student_id = None

        header = QLabel("Students")
        header.setObjectName("pageTitle")
        subtitle = QLabel("Enroll faces, maintain your directory, and keep recognition data current.")
        subtitle.setObjectName("pageSubtitle")

        # ---- Registration form ----
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Full name")
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("Roll number / unique student ID")
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("e.g. MCA or MBA")

        form = QFormLayout()
        form.addRow("Name:", self.name_input)
        form.addRow("Roll number:", self.id_input)
        form.addRow("Class:", self.class_input)

        self.preview_label = QLabel("Capture preview will appear here.")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(360, 270)
        self.preview_label.setObjectName("videoPreview")

        self.capture_progress_label = QLabel(f"Captured 0 / {CAPTURE_COUNT} frames")
        self.capture_progress_label.setObjectName("mutedLabel")

        self.start_capture_btn = QPushButton("Start camera")
        self.start_capture_btn.clicked.connect(self.start_capture)

        self.grab_frame_btn = QPushButton("Capture Frame")
        self.grab_frame_btn.clicked.connect(self.grab_frame)
        self.grab_frame_btn.setEnabled(False)

        self.save_btn = QPushButton("Save student")
        self.save_btn.setObjectName("primaryButton")
        self.save_btn.clicked.connect(self.save_person)
        self.save_btn.setEnabled(False)

        self.stop_capture_btn = QPushButton("Stop camera")
        self.stop_capture_btn.clicked.connect(self.stop_capture)
        self.stop_capture_btn.setEnabled(False)

        self.clear_form_btn = QPushButton("Clear form")
        self.clear_form_btn.clicked.connect(self.clear_form)

        capture_btn_row = QHBoxLayout()
        capture_btn_row.addWidget(self.start_capture_btn)
        capture_btn_row.addWidget(self.grab_frame_btn)
        capture_btn_row.addWidget(self.stop_capture_btn)

        form_box = QGroupBox("Add a new student")
        form_layout = QVBoxLayout()
        form_layout.addLayout(form)
        form_layout.addWidget(self.preview_label)
        form_layout.addWidget(self.capture_progress_label)
        form_layout.addLayout(capture_btn_row)
        row2 = QHBoxLayout()
        row2.addWidget(self.save_btn)
        row2.addWidget(self.clear_form_btn)
        form_layout.addLayout(row2)
        form_box.setLayout(form_layout)
        form_box.setMinimumWidth(430)

        # ---- CRUD table ----
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Roll Number", "Name", "Class", "Actions"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        style_table(self.table)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search by name, roll number, or class")
        self.search_input.textChanged.connect(self.filter_users)
        self.user_count_label = QLabel("0 students")
        self.user_count_label.setObjectName("mutedLabel")

        table_box = QGroupBox("Directory")
        table_layout = QVBoxLayout()
        search_row = QHBoxLayout()
        search_row.addWidget(self.search_input, 1)
        search_row.addWidget(self.user_count_label)
        table_layout.addLayout(search_row)
        table_layout.addWidget(self.table)
        table_box.setLayout(table_layout)

        body = QHBoxLayout()
        body.addWidget(form_box, 1)
        body.addWidget(table_box, 1)

        layout = QVBoxLayout()
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)
        layout.addWidget(header)
        layout.addWidget(subtitle)
        layout.addLayout(body)
        self.setLayout(layout)

        self.refresh_table()

    # ---------------- capture flow ----------------

    def start_capture(self):
        if self.capture_cap is not None:
            return
        self.capture_cap = cv2.VideoCapture(0)
        if not self.capture_cap.isOpened():
            QMessageBox.warning(self, "Camera Error",
                                 "Could not open webcam. If the live recognition camera is "
                                 "running, stop it first (one camera device can't be opened twice).")
            self.capture_cap = None
            return
        self.captured_embeddings = []
        self.capture_progress_label.setText(f"Captured 0 / {CAPTURE_COUNT} frames")
        self.capture_timer.start(30)
        self.start_capture_btn.setEnabled(False)
        self.grab_frame_btn.setEnabled(True)
        self.stop_capture_btn.setEnabled(True)

    def _update_capture_preview(self):
        if self.capture_cap is None:
            return
        ret, frame = self.capture_cap.read()
        if not ret:
            return
        self._last_capture_frame = frame
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
        pixmap = cv_to_qpixmap(frame)
        self.preview_label.setPixmap(
            pixmap.scaled(self.preview_label.width(), self.preview_label.height(), Qt.KeepAspectRatio)
        )

    def grab_frame(self):
        if not hasattr(self, "_last_capture_frame"):
            return
        frame = self._last_capture_frame
        faces = detect_faces(frame)
        if len(faces) == 0:
            QMessageBox.information(self, "No Face Found", "No face detected in the current frame. Try again.")
            return
        # Use the largest detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        embedding = get_face_encoding(frame, (x, y, w, h))
        if embedding is None:
            QMessageBox.information(self, "Encoding Failed", "Could not compute an encoding for this frame. Try again.")
            return

        self.captured_embeddings.append(embedding)
        self.capture_progress_label.setText(f"Captured {len(self.captured_embeddings)} / {CAPTURE_COUNT} frames")

        if len(self.captured_embeddings) >= CAPTURE_COUNT:
            self.grab_frame_btn.setEnabled(False)
            self.save_btn.setEnabled(True)
            QMessageBox.information(self, "Capture Complete",
                                     "Enough frames captured. Click 'Save / Update Person' to store the mean encoding.")

    def stop_capture(self):
        self.capture_timer.stop()
        if self.capture_cap is not None:
            self.capture_cap.release()
            self.capture_cap = None
        self.start_capture_btn.setEnabled(True)
        self.grab_frame_btn.setEnabled(False)
        self.stop_capture_btn.setEnabled(False)
        self.preview_label.setText("Capture preview will appear here.")

    # ---------------- save / CRUD ----------------

    def save_person(self):
        name = self.name_input.text().strip()
        student_id = self.id_input.text().strip()
        class_name = self.class_input.text().strip()

        if not name or not student_id or not class_name:
            QMessageBox.warning(
                self, "Missing Info", "Please enter the student's name, roll number, and class."
            )
            return
        if not self.editing_student_id and not self.captured_embeddings:
            QMessageBox.warning(self, "No Captures", "Please capture at least one face frame first.")
            return

        mean_encoding = (
            np.mean(np.array(self.captured_embeddings), axis=0)
            if self.captured_embeddings else None
        )
        if self.editing_student_id:
            self.db.update_user(
                self.editing_student_id,
                name=name,
                class_name=class_name,
                embedding=mean_encoding,
            )
            detail = "Profile updated."
        else:
            try:
                self.db.add_user(student_id, name, class_name, mean_encoding)
            except sqlite3.IntegrityError:
                QMessageBox.warning(
                    self,
                    "Roll Number Already Exists",
                    f"A student with roll number '{student_id}' is already enrolled.",
                )
                return
            detail = f"Saved face data from {len(self.captured_embeddings)} frame(s)."

        QMessageBox.information(
            self, "Saved", f"{name} ({student_id}) · {class_name}\n{detail}"
        )
        self.clear_form()
        self.refresh_table()
        if self.on_users_changed:
            self.on_users_changed()

    def edit_person(self, student_id, name, class_name):
        self.editing_student_id = student_id
        self.id_input.setText(student_id)
        self.id_input.setEnabled(False)
        self.name_input.setText(name)
        self.class_input.setText(class_name)
        self.captured_embeddings = []
        self.capture_progress_label.setText(
            f"Editing {name}. Update the name only, or capture a new face to replace the saved encoding."
        )
        self.save_btn.setText("Update person")
        self.save_btn.setEnabled(True)

    def delete_person(self, student_id, name):
        confirm = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete '{name}' ({student_id}) and all of their attendance records?",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            self.db.delete_user(student_id)
            if self.editing_student_id == student_id:
                self.clear_form()
            self.refresh_table()
            if self.on_users_changed:
                self.on_users_changed()

    def clear_form(self):
        self.stop_capture()
        self.editing_student_id = None
        self.name_input.clear()
        self.id_input.clear()
        self.class_input.clear()
        self.id_input.setEnabled(True)
        self.captured_embeddings = []
        self.save_btn.setText("Save student")
        self.capture_progress_label.setText(f"Captured 0 / {CAPTURE_COUNT} frames")
        self.save_btn.setEnabled(False)

    def refresh_table(self):
        users = self.db.get_all_users()
        self.table.setRowCount(0)
        for row_idx, (student_id, name, class_name) in enumerate(users):
            self.table.insertRow(row_idx)
            self.table.setItem(row_idx, 0, QTableWidgetItem(student_id))
            self.table.setItem(row_idx, 1, QTableWidgetItem(name))
            self.table.setItem(row_idx, 2, QTableWidgetItem(class_name))

            actions_widget = QWidget()
            actions_layout = QHBoxLayout()
            actions_layout.setContentsMargins(0, 0, 0, 0)

            edit_btn = QPushButton("Edit")
            edit_btn.clicked.connect(
                lambda _, sid=student_id, n=name, c=class_name: self.edit_person(sid, n, c)
            )

            delete_btn = QPushButton("Delete")
            delete_btn.setObjectName("dangerButton")
            delete_btn.clicked.connect(lambda _, sid=student_id, n=name: self.delete_person(sid, n))

            actions_layout.addWidget(edit_btn)
            actions_layout.addWidget(delete_btn)
            actions_widget.setLayout(actions_layout)
            self.table.setCellWidget(row_idx, 3, actions_widget)
        self.user_count_label.setText(f"{len(users)} students")
        self.filter_users(self.search_input.text())

    def filter_users(self, query):
        query = query.strip().casefold()
        for row in range(self.table.rowCount()):
            values = (
                self.table.item(row, 0).text(),
                self.table.item(row, 1).text(),
                self.table.item(row, 2).text(),
            )
            self.table.setRowHidden(row, bool(query) and not any(query in value.casefold() for value in values))


# ======================================================================================
# PAGE 3: ATTENDANCE VISUALIZER & ANALYTICS
# ======================================================================================

class AnalyticsWidget(QWidget):
    def __init__(self, db: Database):
        super().__init__()
        self.db = db

        header = QLabel("Attendance overview")
        header.setObjectName("pageTitle")
        subtitle = QLabel("Review attendance activity and participation across your directory.")
        subtitle.setObjectName("pageSubtitle")

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_all)

        title_row = QHBoxLayout()
        title_row.addWidget(header)
        title_row.addStretch()
        title_row.addWidget(refresh_btn)

        subtitle_row = QHBoxLayout()
        subtitle_row.addWidget(subtitle)
        subtitle_row.addStretch()

        self.metric_values = {}
        metric_row = QHBoxLayout()
        metric_row.setSpacing(12)
        for key, title in (
            ("people", "REGISTERED PEOPLE"),
            ("today", "PRESENT TODAY"),
            ("rate", "TODAY'S ATTENDANCE"),
        ):
            card = QFrame()
            card.setObjectName("metricCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(16, 12, 16, 12)
            card_layout.setSpacing(4)
            label = QLabel(title)
            label.setObjectName("mutedLabel")
            label.setStyleSheet("font-size: 9pt; font-weight: 700;")
            value = QLabel("—")
            value.setStyleSheet("font-size: 22pt; font-weight: 700; color: #20304b;")
            card_layout.addWidget(label)
            card_layout.addWidget(value)
            self.metric_values[key] = value
            metric_row.addWidget(card, 1)

        self.log_table = QTableWidget(0, 6)
        self.log_table.setHorizontalHeaderLabels(
            ["Roll Number", "Name", "Class", "Date", "First Seen", "Last Seen"]
        )
        self.log_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.log_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.log_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.log_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.log_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.log_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        style_table(self.log_table)

        self.percentage_table = QTableWidget(0, 6)
        self.percentage_table.setHorizontalHeaderLabels(
            ["Roll Number", "Name", "Class", "Days Present", "Total Days Tracked", "Presence %"]
        )
        self.percentage_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        style_table(self.percentage_table)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter by name, roll number, or class")
        self.search_input.textChanged.connect(self.filter_tables)

        log_box = QGroupBox("Attendance log")
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.search_input)
        log_layout.addWidget(self.log_table)
        log_box.setLayout(log_layout)

        pct_box = QGroupBox("Participation by person")
        pct_layout = QVBoxLayout()
        pct_layout.addWidget(self.percentage_table)
        pct_box.setLayout(pct_layout)

        layout = QVBoxLayout()
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(12)
        layout.addLayout(title_row)
        layout.addLayout(subtitle_row)
        layout.addLayout(metric_row)
        layout.addWidget(log_box, 3)
        layout.addWidget(pct_box, 2)
        self.setLayout(layout)

        self.refresh_all()

    def refresh_all(self):
        rows = self.db.get_attendance()
        percentages = self.db.get_attendance_percentage()
        people_count = len(self.db.get_all_users())
        today = date.today().isoformat()
        present_today = len({row[0] for row in self.db.get_attendance(today, today)})
        attendance_rate = present_today / people_count * 100 if people_count else 0
        self.metric_values["people"].setText(str(people_count))
        self.metric_values["today"].setText(str(present_today))
        self.metric_values["rate"].setText(f"{attendance_rate:.0f}%")

        self.log_table.setRowCount(0)
        for i, (sid, name, class_name, d, first_seen, last_seen) in enumerate(rows):
            self.log_table.insertRow(i)
            self.log_table.setItem(i, 0, QTableWidgetItem(sid))
            self.log_table.setItem(i, 1, QTableWidgetItem(name or "Unknown"))
            self.log_table.setItem(i, 2, QTableWidgetItem(class_name or "Unassigned"))
            self.log_table.setItem(i, 3, QTableWidgetItem(d))
            self.log_table.setItem(i, 4, QTableWidgetItem(first_seen))
            self.log_table.setItem(i, 5, QTableWidgetItem(last_seen))

        self.percentage_table.setRowCount(0)
        for i, (sid, name, class_name, present_days, total_days, pct) in enumerate(percentages):
            self.percentage_table.insertRow(i)
            self.percentage_table.setItem(i, 0, QTableWidgetItem(sid))
            self.percentage_table.setItem(i, 1, QTableWidgetItem(name))
            self.percentage_table.setItem(i, 2, QTableWidgetItem(class_name))
            self.percentage_table.setItem(i, 3, QTableWidgetItem(str(present_days)))
            self.percentage_table.setItem(i, 4, QTableWidgetItem(str(total_days)))
            self.percentage_table.setItem(i, 5, QTableWidgetItem(f"{pct}%"))
        self.filter_tables(self.search_input.text())

    def filter_tables(self, query):
        query = query.strip().casefold()
        for table in (self.log_table, self.percentage_table):
            for row in range(table.rowCount()):
                student_id = table.item(row, 0).text()
                name = table.item(row, 1).text()
                class_name = table.item(row, 2).text()
                table.setRowHidden(
                    row,
                    bool(query) and not any(
                        query in value.casefold() for value in (student_id, name, class_name)
                    ),
                )


# ======================================================================================
# PAGE 4: EXCEL EXPORT MANAGER
# ======================================================================================

class ExportWidget(QWidget):
    def __init__(self, db: Database):
        super().__init__()
        self.db = db

        header = QLabel("Export reports")
        header.setObjectName("pageTitle")
        subtitle = QLabel(
            "Daily attendance is saved automatically to class-wise Excel sheets. "
            "You can also export a custom date range."
        )
        subtitle.setObjectName("pageSubtitle")

        self.range_combo = QComboBox()
        self.range_combo.addItems(["Weekly (last 7 days)", "Monthly (current month)", "Yearly (current year)"])
        self.range_combo.currentIndexChanged.connect(self.update_summary)

        self.export_btn = QPushButton("Export to Excel")
        self.export_btn.setObjectName("primaryButton")
        self.export_btn.clicked.connect(self.export_excel)

        form = QFormLayout()
        form.setVerticalSpacing(12)
        form.addRow("Reporting period", self.range_combo)

        summary_box = QGroupBox("Report preview")
        summary_layout = QVBoxLayout()
        self.date_range_label = QLabel("")
        self.date_range_label.setObjectName("mutedLabel")
        self.record_count_label = QLabel("—")
        self.record_count_label.setStyleSheet("font-size: 22pt; font-weight: 700; color: #20304b;")
        records_caption = QLabel("attendance records in this period")
        records_caption.setObjectName("mutedLabel")
        summary_layout.addWidget(self.date_range_label)
        summary_layout.addWidget(self.record_count_label)
        summary_layout.addWidget(records_caption)
        summary_box.setLayout(summary_layout)

        self.status_label = QLabel("Choose a reporting period to preview the records.")
        self.status_label.setObjectName("mutedLabel")
        self.status_label.setWordWrap(True)

        layout = QVBoxLayout()
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)
        layout.addWidget(header)
        layout.addWidget(subtitle)
        controls_box = QGroupBox("Report settings")
        controls_box.setLayout(form)
        controls_box.setMaximumWidth(680)
        layout.addWidget(controls_box)
        summary_box.setMaximumWidth(680)
        layout.addWidget(summary_box)
        self.export_btn.setMaximumWidth(220)
        layout.addWidget(self.export_btn)
        layout.addWidget(self.status_label)
        layout.addStretch()
        self.setLayout(layout)
        self.update_summary()

    def _get_date_range(self):
        today = date.today()
        choice = self.range_combo.currentText()
        if choice.startswith("Weekly"):
            start = today - timedelta(days=7)
            end = today
        elif choice.startswith("Monthly"):
            start = today.replace(day=1)
            end = today
        else:  # Yearly
            start = today.replace(month=1, day=1)
            end = today
        return start.isoformat(), end.isoformat()

    def export_excel(self):
        try:
            import pandas as pd
        except ImportError:
            QMessageBox.critical(self, "Missing Dependency",
                                  "pandas (and openpyxl) are required for Excel export.\n"
                                  "Install them with: pip install pandas openpyxl")
            return

        start, end = self._get_date_range()
        rows = self.db.get_attendance(start, end)

        if not rows:
            QMessageBox.information(self, "No Data", "No attendance records found for the selected timeframe.")
            return

        df = pd.DataFrame(
            rows,
            columns=["Roll Number", "Name", "Class", "Date", "First Seen", "Last Seen"],
        )

        default_name = f"attendance_{start}_to_{end}.xlsx"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Attendance Export", default_name, "Excel Files (*.xlsx)")
        if not file_path:
            return

        try:
            df.to_excel(file_path, index=False, engine="openpyxl")
            self.status_label.setText(f"Exported {len(df)} record(s) to {file_path}")
            QMessageBox.information(self, "Export Complete", f"Exported {len(df)} record(s) successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not save the file:\n{e}")

    def update_summary(self):
        start, end = self._get_date_range()
        rows = self.db.get_attendance(start, end)
        self.date_range_label.setText(f"{start}  —  {end}")
        self.record_count_label.setText(str(len(rows)))
        self.export_btn.setEnabled(bool(rows))
        if rows:
            self.status_label.setText("The preview updates when you change the reporting period.")
        else:
            self.status_label.setText("No attendance records were found for this period.")


# ======================================================================================
# MAIN WINDOW (Branded desktop application shell)
# ======================================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Attendance | Admin Console")
        self.setMinimumSize(1120, 720)
        self.resize(1360, 860)

        self.db = Database()

        # ---- Sidebar ----
        self.sidebar = QListWidget()
        self.sidebar.setObjectName("navigation")
        self.sidebar.setFixedWidth(238)
        for label in ["Take attendance", "Students", "Analytics", "Export reports"]:
            QListWidgetItem(label, self.sidebar)
        self.sidebar.currentRowChanged.connect(self.switch_page)

        brand_mark = QLabel("SA")
        brand_mark.setAlignment(Qt.AlignCenter)
        brand_mark.setFixedSize(38, 38)
        brand_mark.setStyleSheet(
            "background: #315fce; color: white; border-radius: 10px; font-size: 13pt; font-weight: 700;"
        )
        brand_name = QLabel("Smart Attendance")
        brand_name.setStyleSheet("color: #ffffff; font-size: 12pt; font-weight: 700;")
        brand_subtitle = QLabel("ADMIN CONSOLE")
        brand_subtitle.setStyleSheet("color: #8fa0ba; font-size: 8pt; font-weight: 700; letter-spacing: 1px;")
        brand_copy = QVBoxLayout()
        brand_copy.setSpacing(1)
        brand_copy.addWidget(brand_name)
        brand_copy.addWidget(brand_subtitle)
        brand_row = QHBoxLayout()
        brand_row.setContentsMargins(18, 20, 12, 20)
        brand_row.setSpacing(10)
        brand_row.addWidget(brand_mark)
        brand_row.addLayout(brand_copy)

        sidebar_panel = QFrame()
        sidebar_panel.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar_panel)
        sidebar_layout.setContentsMargins(0, 0, 0, 12)
        sidebar_layout.setSpacing(4)
        sidebar_layout.addLayout(brand_row)
        nav_caption = QLabel("WORKSPACE")
        nav_caption.setStyleSheet(
            "color: #8292ab; font-size: 8pt; font-weight: 700; letter-spacing: 1px; padding: 8px 18px;"
        )
        sidebar_layout.addWidget(nav_caption)
        sidebar_layout.addWidget(self.sidebar, 1)
        sidebar_panel.setFixedWidth(238)

        # ---- Stacked pages (created once, kept alive so background tasks persist) ----
        self.live_widget = LiveRecognitionWidget(self.db)
        self.registration_widget = RegistrationWidget(self.db, on_users_changed=self.on_users_changed)
        self.analytics_widget = AnalyticsWidget(self.db)
        self.export_widget = ExportWidget(self.db)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.live_widget)
        self.stack.addWidget(self.registration_widget)
        self.stack.addWidget(self.analytics_widget)
        self.stack.addWidget(self.export_widget)

        # ---- Top bar ----
        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_bar_layout = QHBoxLayout()
        top_bar_layout.setContentsMargins(22, 8, 22, 8)
        top_bar_layout.setSpacing(10)
        app_title = QLabel("Attendance management")
        app_title.setStyleSheet("font-size: 11pt; font-weight: 700;")
        top_bar_layout.addWidget(app_title)
        top_bar_layout.addStretch()

        add_user_btn = QPushButton("Add new student")
        add_user_btn.setToolTip("Add a student with name, roll number, class, and face capture")
        add_user_btn.clicked.connect(self.add_new_user)
        top_bar_layout.addWidget(add_user_btn)

        attendance_btn = QPushButton("Take attendance")
        attendance_btn.setObjectName("primaryButton")
        attendance_btn.setToolTip("Open the live attendance camera")
        attendance_btn.clicked.connect(self.start_attendance)
        top_bar_layout.addWidget(attendance_btn)

        user_badge = QLabel("ADMIN  ·  admin")
        user_badge.setStyleSheet(
            "background: #eef2f8; color: #43536c; border-radius: 8px; padding: 8px 10px; font-weight: 700;"
        )
        top_bar_layout.addWidget(user_badge)
        top_bar.setLayout(top_bar_layout)
        top_bar.setFixedHeight(64)

        # ---- Assemble layout ----
        body_layout = QHBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)
        body_layout.addWidget(sidebar_panel)
        body_layout.addWidget(self.stack, 1)

        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(16, 16, 16, 16)
        outer_layout.setSpacing(12)
        outer_layout.addWidget(top_bar)
        outer_layout.addLayout(body_layout)

        central = QWidget()
        central.setLayout(outer_layout)
        self.setCentralWidget(central)

        self.sidebar.setCurrentRow(0)

    def switch_page(self, index):
        self.stack.setCurrentIndex(index)
        if index == 2:
            self.analytics_widget.refresh_all()
        elif index == 1:
            self.registration_widget.refresh_table()
        elif index == 3:
            self.export_widget.update_summary()

    def add_new_user(self):
        self.live_widget.stop_camera()
        self.sidebar.setCurrentRow(1)
        self.registration_widget.start_capture()

    def start_attendance(self):
        self.registration_widget.stop_capture()
        self.sidebar.setCurrentRow(0)
        self.live_widget.start_camera()

    def on_users_changed(self):
        """Keep a running camera thread's in-RAM embeddings in sync after CRUD changes."""
        self.live_widget.refresh_known_faces()
        self.analytics_widget.refresh_all()

    def closeEvent(self, event):
        self.live_widget.stop_camera()
        self.registration_widget.stop_capture()
        self.db.conn.close()
        event.accept()


# ======================================================================================
# ENTRY POINT
# ======================================================================================

def main():
    if not ADMIN_USERNAME or not ADMIN_PASSWORD:
        print(
            "[ERROR] Set ATTENDANCE_ADMIN_USERNAME and ATTENDANCE_ADMIN_PASSWORD "
            "in your local .env file before starting the desktop application.",
            file=sys.stderr,
        )
        return 1

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(APP_STYLESHEET)

    if not HAS_FACE_RECOGNITION:
        print("[INFO] 'face_recognition' unavailable - using a built-in pixel-based "
              "fallback encoder so the app still runs. For real face recognition accuracy, "
              "run: pip install face_recognition")

    if not HAS_MEDIAPIPE:
        print("[INFO] 'mediapipe' not installed - liveness (blink) detection is disabled "
              "and faces will be matched without a liveness check. Run: pip install mediapipe")

    login = LoginDialog()
    if login.exec_() != QDialog.Accepted or not login.authenticated:
        sys.exit(0)

    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
