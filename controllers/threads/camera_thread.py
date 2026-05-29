import time
import queue
import logging
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, THREAD_SLEEP

logger = logging.getLogger(__name__)

# Đọc frame từ camera và phân phối cho 2 pipeline
class CameraThread(QThread):
    error_occurred = pyqtSignal(str)

    def __init__(self, stream_queue, ai_queue):
        super().__init__()
        self.stream_queue = stream_queue
        self.ai_queue = ai_queue
        self.running = False

    def run(self):
        self.running = True
        if isinstance(CAMERA_INDEX, int):
            # Webcam USB local
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(CAMERA_INDEX)
        else:
            # IP Camera (RTSP/HTTP Stream)
            cap = cv2.VideoCapture(CAMERA_INDEX)

        if not cap.isOpened():
            self.error_occurred.emit(f"Không mở được camera (index={CAMERA_INDEX})")
            return

        # Ép thông số cho webcam
        if isinstance(CAMERA_INDEX, int):
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            
        # BUFFERSIZE = 1 rất quan trọng cho IP Camera để không bị delay (tích tụ frame cũ)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        logger.info(f"Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Không đọc được frame, thử lại…")
                time.sleep(THREAD_SLEEP)
                continue

            # Push to StreamQueue
            if self.stream_queue.full():
                try: self.stream_queue.get_nowait()
                except queue.Empty: pass
            try: self.stream_queue.put_nowait(frame)
            except queue.Full: pass

            # Push to AI Queue
            if self.ai_queue.full():
                try: self.ai_queue.get_nowait()
                except queue.Empty: pass
            try: self.ai_queue.put_nowait(frame)
            except queue.Full: pass

        cap.release()
        logger.info("CameraThread stopped.")

    def stop(self):
        self.running = False
