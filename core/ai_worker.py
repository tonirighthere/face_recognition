"""
core/ai_worker.py — QThread xử lý camera + AI pipeline
Signal flow:
  frame_ready(np.ndarray)         → liveview widget cập nhật ảnh
  recognition_result(list[dict])  → liveview widget cập nhật danh sách
  status_changed(str)             → status bar
  error_occurred(str)             → dialog lỗi
  model_loaded()                  → UI kích hoạt nút Start
"""

import logging
import time
from typing import List, Dict, Any

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, FRAME_SKIP, SIMILARITY_THRESHOLD
from core.face_engine import FaceEngine
from core.vector_store import VectorStore
from core.tracker import SimpleTracker

logger = logging.getLogger(__name__)

# Màu bbox (BGR)
COLOR_KNOWN   = (0, 230, 100)   # Xanh lá nhận diện được
COLOR_UNKNOWN = (80, 80, 255)   # Đỏ chưa nhận diện
COLOR_TEXT_BG = (20, 20, 20)


class AIWorker(QThread):
    frame_ready        = pyqtSignal(np.ndarray)
    recognition_result = pyqtSignal(list)
    status_changed     = pyqtSignal(str)
    error_occurred     = pyqtSignal(str)
    model_loaded       = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running  = False
        self._paused   = False
        self._engine   = FaceEngine.instance()
        self._vstore   = VectorStore.instance()
        self._tracker  = SimpleTracker()
        self._frame_count = 0
        self._fps      = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def start_stream(self):
        self._running = True
        self._paused  = False
        if not self.isRunning():
            self.start()

    def stop_stream(self):
        self._running = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    # ── Thread main ───────────────────────────────────────────────────────────

    def run(self):
        # 1. Load model (lần đầu)
        if not self._engine.is_loaded:
            self.status_changed.emit("Đang tải model AI…")
            try:
                self._engine.load()
            except Exception as e:
                self.error_occurred.emit(f"Lỗi load model: {e}")
                return

        # 1.5 Load vector database
        from database.db_manager import DatabaseManager
        self._vstore.load_from_db(DatabaseManager())
        
        self.model_loaded.emit()

        # 2. Mở camera
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            self.error_occurred.emit(f"Không mở được camera (index={CAMERA_INDEX})")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._tracker.reset()
        self.status_changed.emit("Camera đang chạy…")

        fps_start_time = time.time()
        fps_frames = 0

        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret:
                logger.warning("Không đọc được frame, thử lại…")
                time.sleep(0.1)
                continue

            self._frame_count += 1
            fps_frames += 1

            # 3. Tính FPS (Cập nhật mỗi 1 giây)
            now = time.time()
            if now - fps_start_time >= 1.0:
                self._fps = fps_frames / (now - fps_start_time)
                fps_frames = 0
                fps_start_time = now

            # 4. Chạy AI mỗi FRAME_SKIP frame
            if self._frame_count % FRAME_SKIP == 0:
                detections = self._engine.detect(frame)
                tracks     = self._tracker.update(detections)

                results: List[Dict[str, Any]] = []
                for track in tracks:
                    if track.lost > 0:
                        continue
                    # Chỉ re-embed nếu chưa nhận diện
                    if not track.recognized:
                        emb = self._engine.get_embedding(frame, track.bbox)
                        if emb is not None:
                            hit = self._vstore.search_best(emb)
                            if hit:
                                track.person_id   = hit[0]
                                track.person_name = hit[1]
                                track.similarity  = hit[2]
                                track.recognized  = True

                    results.append({
                        "track_id":    track.track_id,
                        "bbox":        track.bbox,
                        "person_id":   track.person_id,
                        "person_name": track.person_name,
                        "similarity":  track.similarity,
                        "recognized":  track.recognized,
                    })

                self.recognition_result.emit(results)

            # 5. Vẽ annotations
            annotated = self._draw(frame.copy(), self._tracker.active_tracks)

            # 6. Phát frame ra UI
            self.frame_ready.emit(annotated)

        cap.release()
        self.status_changed.emit("Camera đã dừng.")

    # ── Draw ──────────────────────────────────────────────────────────────────

    def _draw(self, frame: np.ndarray, tracks) -> np.ndarray:
        h, w = frame.shape[:2]

        for track in tracks:
            x1, y1, x2, y2, conf = track.bbox
            color = COLOR_KNOWN if track.recognized else COLOR_UNKNOWN

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label
            if track.recognized and track.person_name:
                label = f"{track.person_name}  {track.similarity:.0%}"
            else:
                label = f"Không rõ  #{track.track_id}"

            # Background cho text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            lx, ly = x1, max(y1 - 4, th + 4)
            cv2.rectangle(frame, (lx, ly - th - 6), (lx + tw + 8, ly + 2), color, -1)
            cv2.putText(frame, label, (lx + 4, ly - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA)

        # FPS overlay
        cv2.putText(frame, f"FPS: {int(self._fps)}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 100), 2, cv2.LINE_AA)
        return frame
