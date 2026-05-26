import time
import queue
import logging
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT

logger = logging.getLogger(__name__)


class CameraThread(QThread):
    error_occurred = pyqtSignal(str)

    def __init__(self, stream_queue, ai_queue):
        super().__init__()
        self.stream_queue = stream_queue
        self.ai_queue = ai_queue
        self.running = False

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            self.error_occurred.emit(f"Không mở được camera (index={CAMERA_INDEX})")
            return

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        real_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {real_fps}FPS")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Không đọc được frame, thử lại…")
                time.sleep(0.05)
                continue

            # --- Push to StreamQueue (Display - 30 FPS) ---
            if self.stream_queue.full():
                try: self.stream_queue.get_nowait()
                except queue.Empty: pass
            try: self.stream_queue.put(frame)
            except: pass

            # --- Push to AI Queue (AI Pipeline) ---
            if self.ai_queue.full():
                try: self.ai_queue.get_nowait()
                except queue.Empty: pass
            try: self.ai_queue.put(frame)
            except: pass

        cap.release()
        logger.info("CameraThread stopped.")

    def stop(self):
        self.running = False
