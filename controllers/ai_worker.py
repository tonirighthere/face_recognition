"""
controllers/ai_worker.py — Quản lý 5-thread asynchronous AI pipeline
"""

import logging
import time
import queue
from PyQt5.QtCore import QThread, pyqtSignal

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.face_engine import FaceEngine
from models.vector_store import VectorStore
from core.tracker import SimpleTracker
from controllers.threads import CameraThread, DetectThread, TrackThread, FaceThread, StreamThread
from config import QUEUE_MAXSIZE

logger = logging.getLogger(__name__)


import threading

class AIWorker(QThread):
    """
    Manager điều phối toàn bộ pipeline theo mô hình Decoupled:
    - Nhánh 1 (Display): Camera -> StreamThread -> UI (30 FPS)
    - Nhánh 2 (AI): Camera -> Detect -> Track -> Face -> Cập nhật Shared State (chậm hơn nhưng không làm drop frame Display)
    """
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    model_loaded   = pyqtSignal()

    _stream_thread: StreamThread = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._paused  = False

        self._engine  = FaceEngine.instance()
        self._vstore  = VectorStore.instance()
        self._tracker = SimpleTracker()

        # Shared state giữa AI pipeline và Display pipeline
        self.shared_state = {
            "tracks_info":    [],
            "display_frame":  None,
            "lock":           threading.Lock()
        }

        # Tạo stream_thread sớm
        self.stream_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._stream_thread = StreamThread(self.stream_queue, self.shared_state)

    @property
    def stream_thread(self) -> StreamThread:
        return self._stream_thread

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

    def run(self):
        # 1. Load model
        if not self._engine.is_loaded:
            self.status_changed.emit("Đang tải model AI…")
            try:
                self._engine.load()
            except Exception as e:
                self.error_occurred.emit(f"Lỗi load model: {e}")
                return

        # 2. Load vector DB
        from models.db_manager import DatabaseManager
        self._vstore.load_from_db(DatabaseManager())
        self.model_loaded.emit()

        # 3. Khởi tạo Queue và Thread
        ai_queue     = queue.Queue(maxsize=QUEUE_MAXSIZE)
        detect_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        track_queue  = queue.Queue(maxsize=QUEUE_MAXSIZE)

        cam_thread    = CameraThread(self.stream_queue, ai_queue)
        detect_thread = DetectThread(ai_queue, detect_queue, self._engine)
        track_thread  = TrackThread(detect_queue, track_queue, self._tracker)
        face_thread   = FaceThread(track_queue, self.shared_state, self._engine, self._vstore)
        stream_thread = self._stream_thread

        # 4. Kết nối error signal
        cam_thread.error_occurred.connect(self.error_occurred)

        # 5. Kích hoạt tất cả
        self.status_changed.emit("Camera đang chạy")
        cam_thread.start()
        detect_thread.start()
        track_thread.start()
        face_thread.start()
        stream_thread.start()
        logger.info("All 5 threads started.")

        # 6. Vòng lặp chờ stop
        while self._running:
            detect_thread.paused = self._paused
            track_thread.paused  = self._paused
            face_thread.paused   = self._paused
            time.sleep(0.05)

        # 7. Dọn dẹp
        cam_thread.stop()
        detect_thread.stop()
        track_thread.stop()
        face_thread.stop()
        stream_thread.stop()

        cam_thread.wait()
        detect_thread.wait()
        track_thread.wait()
        face_thread.wait()
        stream_thread.wait()

        self._tracker.reset()
        self.status_changed.emit("Camera đã dừng.")
        logger.info("AIWorker stopped cleanly.")
