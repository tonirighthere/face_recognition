"""
Face Detection (MediaPipe) + Quality Filter + Embedding (InsightFace ArcFace)

Pipeline:
  MediaPipe detect → Quality filter (blur / roll / yaw / pitch) → InsightFace embed

Các khuôn mặt không đủ chất lượng bị loại TRƯỚC khi vào InsightFace,
giúp giảm tải đáng kể khi chỉ sử dụng CPU.
"""

import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp   # import một lần duy nhất khi khởi động module

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODEL_DIR,
    MP_MIN_DETECTION_CONFIDENCE,
    INSIGHTFACE_MODEL, INSIGHTFACE_DEVICE,
)
from utils.face_utils import is_face_valid

logger = logging.getLogger(__name__)

# BBox type: (x1, y1, x2, y2, conf)
BBox = Tuple[int, int, int, int, float]


class FaceEngine:
    """
    Kết hợp MediaPipe (detect + quality filter) và InsightFace ArcFace (embed).
    Sử dụng như singleton qua FaceEngine.instance().
    """

    _instance: Optional["FaceEngine"] = None

    @classmethod
    def instance(cls) -> "FaceEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._mp_detector  = None   # mediapipe FaceDetection
        self._app          = None   # InsightFace FaceAnalysis
        self._loaded       = False

    def load(self):
        """Tải cả hai model."""
        if self._loaded:
            return
        # Load InsightFace (onnxruntime) TRƯỚC để tránh xung đột DLL trên Windows
        self._load_insightface()
        self._load_mediapipe()
        self._loaded = True
        logger.info("FaceEngine loaded successfully (MediaPipe + InsightFace).")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ─── MediaPipe Detection ───────────────────────────────────────────────────

    def _load_mediapipe(self):
        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions

            model_path = MODEL_DIR / "blaze_face_short_range.tflite"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"MediaPipe model not found at {model_path}. "
                    "Chạy lệnh sau để tải về:\n"
                    "  python -c \"import urllib.request; "
                    "urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/"
                    "face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite', "
                    "'weights/blaze_face_short_range.tflite')\""
                )

            options = FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                min_detection_confidence=MP_MIN_DETECTION_CONFIDENCE,
            )
            self._mp_detector = FaceDetector.create_from_options(options)
            logger.info(
                f"MediaPipe FaceDetector (Tasks API) loaded: {model_path.name}, "
                f"conf>={MP_MIN_DETECTION_CONFIDENCE}"
            )
        except Exception as e:
            logger.error(f"Cannot load MediaPipe: {e}")
            raise


    def detect(self, frame: np.ndarray, apply_filter: bool = True) -> List[BBox]:
        """
        Detect khuôn mặt trong frame bằng MediaPipe Tasks API.
        Nếu apply_filter=True, mỗi khuôn mặt được qua bộ lọc chất lượng (blur / roll / yaw / pitch).
        Trả về list (x1, y1, x2, y2, conf).
        """
        if self._mp_detector is None:
            return []

        h, w = frame.shape[:2]

        # Tasks API dùng mediapipe.Image (RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._mp_detector.detect(mp_image)

        if not result.detections:
            return []

        boxes: List[BBox] = []
        for detection in result.detections:
            score = detection.categories[0].score
            bb    = detection.bounding_box

            x1 = max(0, bb.origin_x)
            y1 = max(0, bb.origin_y)
            x2 = min(w, bb.origin_x + bb.width)
            y2 = min(h, bb.origin_y + bb.height)

            if x2 <= x1 or y2 <= y1:
                continue

            # Keypoints pixel: [right_eye, left_eye, nose_tip, mouth_center, right_ear, left_ear]
            kp_px = [
                (int(k.x * w), int(k.y * h))
                for k in detection.keypoints
            ]

            if apply_filter and not is_face_valid(frame, (x1, y1, x2, y2, score), kp_px):
                logger.debug(
                    f"Face filtered out at ({x1},{y1},{x2},{y2}) score={score:.2f}"
                )
                continue

            boxes.append((x1, y1, x2, y2, float(score)))

        return boxes

    # InsightFace Embedding

    def _load_insightface(self):
        try:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(
                name=INSIGHTFACE_MODEL,
                root=str(MODEL_DIR),
                providers=["CPUExecutionProvider"] if INSIGHTFACE_DEVICE < 0
                          else ["CUDAExecutionProvider"],
            )
            self._app.prepare(ctx_id=INSIGHTFACE_DEVICE, det_size=(320, 320))
            logger.info(f"InsightFace loaded: {INSIGHTFACE_MODEL}")
        except Exception as e:
            logger.error(f"Cannot load InsightFace: {e}")
            raise

    def get_embedding(self, frame: np.ndarray, bbox: BBox) -> Optional[np.ndarray]:
        """
        Crop khuôn mặt từ bbox (có padding) → align → embed.
        Trả về vector 512D float32 đã L2-normalize, hoặc None nếu lỗi.
        """
        if self._app is None:
            return None
        x1, y1, x2, y2, _ = bbox
        h, w = frame.shape[:2]

        # Mở rộng bbox (padding ~30%) để InsightFace dễ detect 5 landmarks
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(bw * 0.3), int(bh * 0.3)

        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(w, x2 + pad_x)
        py2 = min(h, y2 + pad_y)

        face_crop = frame[py1:py2, px1:px2]
        if face_crop.size == 0:
            return None

        # InsightFace: Detect → Align → Embed trên vùng crop
        faces = self._app.get(face_crop)
        if not faces:
            return None

        # Lấy khuôn mặt lớn nhất (thường là khuôn mặt chính)
        faces = sorted(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )

        emb = faces[0].normed_embedding   # đã normalize L2
        return emb.astype(np.float32)

    # ─── CRUD helpers ──────────────────────────────────────────────────────────

    def get_embedding_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Embed từ ảnh tĩnh — dùng khi đăng ký (CRUD) khuôn mặt mới vào DB.
        Sử dụng MediaPipe để detect (bỏ qua filter để chấp nhận ảnh selfie/chỉnh sửa) → InsightFace embed.
        Trả về None nếu không phát hiện đúng 1 khuôn mặt.
        """
        bboxes = self.detect(image, apply_filter=False)
        if len(bboxes) != 1:
            return None
        return self.get_embedding(image, bboxes[0])

    def count_faces(self, image: np.ndarray) -> int:
        """
        Đếm số khuôn mặt trong ảnh tĩnh (bỏ qua filter).
        """
        return len(self.detect(image, apply_filter=False))
