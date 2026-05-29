import time
import queue
import logging
import traceback
from PyQt5.QtCore import QThread

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import QUEUE_TIMEOUT, THREAD_SLEEP

logger = logging.getLogger(__name__)


class DetectThread(QThread):
    def __init__(self, in_queue, out_queue, engine):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.engine = engine
        self.running = False
        self.paused = False
        self.frame_count = 0

    def run(self):
        self.running = True
        logger.info("DetectThread started.")

        while self.running:
            if self.paused:
                time.sleep(THREAD_SLEEP)
                continue

            # Lấy frame từ AI Queue
            try:
                frame = self.in_queue.get(timeout=QUEUE_TIMEOUT)
            except queue.Empty:
                continue

            # Chạy AI trên frame mới nhất
            try:
                detections = self.engine.detect(frame)
            except Exception as e:
                logger.error(f"DetectThread MediaPipe Error: {e}")
                detections = []

            # Chuyển sang TrackThread
            if self.out_queue.full():
                try: self.out_queue.get_nowait()
                except queue.Empty: pass
            self.out_queue.put((frame, detections, True))

        logger.info("DetectThread stopped.")

    def stop(self):
        self.running = False
