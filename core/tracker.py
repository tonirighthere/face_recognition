"""
core/tracker.py — Simple IoU-based face tracker (không cần thư viện ngoài)
Giữ track_id ổn định qua các frame để tránh re-embed liên tục.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TRACK_MAX_LOST

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int, float]   # x1, y1, x2, y2, conf


class Track:
    _next_id = 1

    def __init__(self, bbox: BBox):
        self.track_id = Track._next_id
        Track._next_id += 1
        self.bbox = bbox
        self.lost = 0
        self.hits = 1
        # Cache nhận diện để không re-search mỗi frame
        self.person_id:   Optional[int]   = None
        self.person_name: Optional[str]   = None
        self.similarity:  float           = 0.0
        self.recognized:  bool            = False

    def update(self, bbox: BBox):
        self.bbox = bbox
        self.lost = 0
        self.hits += 1


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2, _ = a
    bx1, by1, bx2, by2, _ = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class SimpleTracker:
    """
    Hungarian-free IoU tracker — đủ dùng cho liveview khuôn mặt tốc độ cao.
    """

    def __init__(self, iou_threshold: float = 0.35, max_lost: int = TRACK_MAX_LOST):
        self._tracks: Dict[int, Track] = {}
        self._iou_threshold = iou_threshold
        self._max_lost = max_lost

    def update(self, detections: List[BBox]) -> List[Track]:
        """
        Cập nhật tracker với list bboxes phát hiện mới.
        Trả về list Track còn sống (kể cả đang lost ngắn hạn).
        """
        matched_track_ids = set()

        for det in detections:
            best_tid = None
            best_iou = self._iou_threshold
            for tid, track in self._tracks.items():
                if tid in matched_track_ids:
                    continue
                score = _iou(track.bbox, det)
                if score > best_iou:
                    best_iou = score
                    best_tid = tid

            if best_tid is not None:
                self._tracks[best_tid].update(det)
                matched_track_ids.add(best_tid)
            else:
                new_track = Track(det)
                self._tracks[new_track.track_id] = new_track
                matched_track_ids.add(new_track.track_id)

        # Tăng lost counter cho track không match
        to_delete = []
        for tid, track in self._tracks.items():
            if tid not in matched_track_ids:
                track.lost += 1
                if track.lost > self._max_lost:
                    to_delete.append(tid)

        for tid in to_delete:
            del self._tracks[tid]

        return list(self._tracks.values())

    def reset(self):
        self._tracks.clear()

    @property
    def active_tracks(self) -> List[Track]:
        return [t for t in self._tracks.values() if t.lost == 0]
