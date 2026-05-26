import time
import queue
import logging
import traceback
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import QThread, pyqtSignal

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.image_utils import get_pil_font, draw_text_pil

logger = logging.getLogger(__name__)

COLOR_KNOWN   = (0, 230, 100)   # Xanh lá — đã nhận diện
COLOR_UNKNOWN = (80, 80, 255)   # Đỏ — chưa nhận diện


class StreamThread(QThread):
    frame_ready        = pyqtSignal(object)
    recognition_result = pyqtSignal(object)

    def __init__(self, stream_queue, shared_state):
        super().__init__()
        self.stream_queue = stream_queue
        self.shared_state = shared_state
        self.running = False

        self._font     = get_pil_font("arial.ttf", 16)
        self._font_fps = get_pil_font("arial.ttf", 20)

    def run(self):
        self.running = True
        fps_t0 = time.time()
        fps_count = 0
        fps = 0.0
        logger.info("StreamThread started.")

        while self.running:
            # --- Lấy Frame Gốc (Real-time từ Camera) ---
            try:
                frame = self.stream_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # --- Lấy Tracking Info mới nhất từ AI ---
            with self.shared_state["lock"]:
                tracks_info = list(self.shared_state["tracks_info"])
            
            # Cập nhật GUI List
            results = [t for t in tracks_info if t["recognized"]]
            self.recognition_result.emit(results)

            # --- Tính FPS Display ---
            fps_count += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps = fps_count / (now - fps_t0)
                fps_count = 0
                fps_t0 = now

            # --- Vẽ ---
            try:
                draw_frame = self._draw(frame, tracks_info, fps)
                self.frame_ready.emit(draw_frame)
            except Exception:
                logger.error("[StreamThread] draw/emit failed:")
                traceback.print_exc()

        logger.info("StreamThread stopped.")

    def stop(self):
        self.running = False

    # ------------------------------------------------------------------
    def _draw(self, frame: np.ndarray, tracks_info: list, fps: float) -> np.ndarray:
        frame = frame.copy()   # không mutate frame gốc (tránh race condition)

        # Vẽ bbox bằng OpenCV
        for t in tracks_info:
            x1, y1, x2, y2, _ = t["bbox"]
            color = COLOR_KNOWN if t["recognized"] else COLOR_UNKNOWN
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Vẽ text bằng PIL
        return draw_text_pil(frame, tracks_info, fps, self._font, self._font_fps, COLOR_KNOWN, COLOR_UNKNOWN)
