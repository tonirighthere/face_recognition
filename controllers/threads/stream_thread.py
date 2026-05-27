import time
import queue
import logging
import traceback
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DISPLAY_FPS
from utils.image_utils import get_pil_font, draw_text_pil

logger = logging.getLogger(__name__)

COLOR_KNOWN   = (0, 230, 100)   # Xanh lá 
COLOR_UNKNOWN = (80, 80, 255)   # Đỏ


class StreamThread(QThread):
    recognition_result = pyqtSignal(object)  # giữ lại cho result panel

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
        last_result_emit = 0.0
        logger.info("StreamThread started.")

        while self.running:
            # Lấy Frame Gốc (Real-time từ Camera)
            try:
                frame = self.stream_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Lấy Tracking Info mới nhất từ AI
            with self.shared_state["lock"]:
                tracks_info = list(self.shared_state["tracks_info"])

            # Cập nhật GUI List tối đa 2 lần/giây (tránh spam signal)
            now = time.time()
            if now - last_result_emit >= 0.5:
                results = [t for t in tracks_info if t["recognized"]]
                self.recognition_result.emit(results)
                last_result_emit = now

            # Tính FPS Display
            fps_count += 1
            now = time.time()
            if now - fps_t0 >= 1.0:
                fps = fps_count / (now - fps_t0)
                fps_count = 0
                fps_t0 = now

            # Vẽ vào shared_state (QTimer sẽ pull)
            try:
                draw_frame = draw_text_pil(frame, tracks_info, fps, self._font, self._font_fps, COLOR_KNOWN, COLOR_UNKNOWN)
                with self.shared_state["lock"]:
                    self.shared_state["display_frame"] = draw_frame
            except Exception:
                logger.error("[StreamThread] draw_text_pil failed:")
                traceback.print_exc()

        logger.info("StreamThread stopped.")

    def stop(self):
        self.running = False
