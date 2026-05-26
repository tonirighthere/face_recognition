import time
import queue
import logging
import traceback
import numpy as np
from PyQt5.QtCore import QThread

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

COOLDOWN   = 1.0
DELTA_CONF = 0.1


class FaceThread(QThread):
    def __init__(self, in_queue, shared_state, engine, vector_store):
        super().__init__()
        self.in_queue     = in_queue
        self.shared_state = shared_state
        self.engine       = engine
        self.vstore       = vector_store
        self.running = False
        self.paused = False

    def run(self):
        self.running = True
        logger.info("FaceThread started.")

        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue

            # --- Lấy từ TrackThread ---
            try:
                frame, tracks, is_ai_frame = self.in_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # --- Nhận diện khuôn mặt ---
            if is_ai_frame:
                tracks_info_copy = []
                try:
                    for track in tracks:
                        if track.lost > 0:
                            continue

                        x1, y1, x2, y2, conf = track.bbox
                        current_area = (x2 - x1) * (y2 - y1)
                        now_time = time.time()

                        needs_re_embed = (
                            not track.recognized
                            or (
                                (current_area > track.best_area * 1.2 or conf > track.best_conf + DELTA_CONF)
                                and (now_time - track.last_embed_time > COOLDOWN)
                            )
                        )

                        if needs_re_embed:
                            new_embedding = self.engine.get_embedding(frame, track.bbox)
                            if new_embedding is not None:
                                track.last_embed_time = now_time
                                hit = self.vstore.search_best(new_embedding)
                                if hit:
                                    new_sim = hit[2]
                                    if not track.recognized or (new_sim > track.similarity and new_sim > SIMILARITY_THRESHOLD):
                                        if track.recognized and track.embedding is not None and track.person_id == hit[0]:
                                            track.embedding = 0.8 * track.embedding + 0.2 * new_embedding
                                            track.embedding /= np.linalg.norm(track.embedding)
                                        else:
                                            track.embedding = new_embedding
                                        track.person_id   = hit[0]
                                        track.person_name = hit[1]
                                        track.similarity  = new_sim
                                        track.recognized  = True
                                        track.best_area   = current_area
                                        track.best_conf   = conf

                        res_dict = {
                            "track_id":    track.track_id,
                            "bbox":        track.bbox,
                            "person_id":   track.person_id,
                            "person_name": track.person_name,
                            "similarity":  track.similarity,
                            "recognized":  track.recognized,
                        }
                        tracks_info_copy.append(res_dict)

                except Exception:
                    logger.error("[FaceThread] recognition failed:")
                    traceback.print_exc()

            # --- Update Shared State ---
            with self.shared_state["lock"]:
                self.shared_state["tracks_info"] = tracks_info_copy

        logger.info("FaceThread stopped.")

    def stop(self):
        self.running = False
