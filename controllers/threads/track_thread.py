import time
import queue
import logging
import traceback
from PyQt5.QtCore import QThread

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
                time.sleep(0.05)
                continue

            # Lấy từ DetectThread
            try:
                frame, detections, is_ai_frame = self.in_queue.get(timeout=0.1)
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
                    tracks = [t for t in last_tracks if current_time - t.last_update_time < 1.0]
            else:
                current_time = time.time()
                tracks = [t for t in last_tracks if current_time - t.last_update_time < 1.0]

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
