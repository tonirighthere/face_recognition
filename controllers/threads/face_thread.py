import time
import queue
import logging
import traceback
import numpy as np
from PyQt5.QtCore import QThread

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    SIMILARITY_THRESHOLD, 
    TRACK_COOLDOWN, 
    TRACK_DELTA_CONF,
    TRACK_AREA_RATIO,
    TRACK_EMA_ALPHA,
    QUEUE_TIMEOUT,
    THREAD_SLEEP
)
from utils.tracking_utils import should_re_embed, update_embedding_ema

logger = logging.getLogger(__name__)


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
                time.sleep(THREAD_SLEEP)
                continue

            # Lấy từ TrackThread
            try:
                frame, tracks, is_ai_frame = self.in_queue.get(timeout=QUEUE_TIMEOUT)
            except queue.Empty:
                continue

            # Nhận diện khuôn mặt
            if is_ai_frame:
                tracks_info_copy = []
                try:
                    for track in tracks:
                        if track.lost > 0:
                            continue

                        x1, y1, x2, y2, conf = track.bbox
                        current_area = (x2 - x1) * (y2 - y1)
                        now_time = time.time()

                        needs_re_embed = should_re_embed(
                            track, 
                            current_area, 
                            conf, 
                            now_time, 
                            TRACK_COOLDOWN, 
                            TRACK_DELTA_CONF,
                            TRACK_AREA_RATIO
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
                                            track.embedding = update_embedding_ema(track.embedding, new_embedding, alpha=TRACK_EMA_ALPHA)
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

            # Update Shared State
            with self.shared_state["lock"]:
                self.shared_state["tracks_info"] = tracks_info_copy

        logger.info("FaceThread stopped.")

    def stop(self):
        self.running = False
