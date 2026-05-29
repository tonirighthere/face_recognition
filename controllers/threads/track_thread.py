import time
import queue
import logging
import traceback
from PyQt5.QtCore import QThread

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import QUEUE_TIMEOUT, THREAD_SLEEP, TRACK_RETAIN_SECONDS

logger = logging.getLogger(__name__)


class TrackThread(QThread):
    def __init__(self, in_queue, out_queue, tracker):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.tracker = tracker
        self.running = False
        self.paused = False

    def run(self):
        self.running = True
        last_tracks = []
        logger.info("TrackThread started.")

        while self.running:
            if self.paused:
                time.sleep(THREAD_SLEEP)
                continue

            # Lấy từ DetectThread
            try:
                frame, detections, is_ai_frame = self.in_queue.get(timeout=QUEUE_TIMEOUT)
            except queue.Empty:
                continue

            # Cập nhật tracker
            if is_ai_frame and detections is not None:
                try:
                    tracks = self.tracker.update(detections)
                    last_tracks = tracks
                except Exception:
                    logger.error("[TrackThread] tracker.update() failed:")
                    traceback.print_exc()
                    current_time = time.time()
                    tracks = [t for t in last_tracks if current_time - t.last_update_time < TRACK_RETAIN_SECONDS]
            else:
                current_time = time.time()
                tracks = [t for t in last_tracks if current_time - t.last_update_time < TRACK_RETAIN_SECONDS]

            # Đẩy xuống FaceThread
            if self.out_queue.full():
                try:
                    self.out_queue.get_nowait()
                except queue.Empty:
                    pass
            self.out_queue.put((frame, tracks, is_ai_frame))

        logger.info("TrackThread stopped.")

    def stop(self):
        self.running = False
