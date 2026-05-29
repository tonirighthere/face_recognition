"""
Face Detection (YOLOv8-face) + Embedding (InsightFace ArcFace)
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODEL_DIR,
    YOLO_MODEL_NAME, YOLO_CONF_THRESH, YOLO_IOU_THRESH, YOLO_IMG_SIZE, YOLO_DEVICE,
    INSIGHTFACE_MODEL, INSIGHTFACE_DEVICE,
)

logger = logging.getLogger(__name__)

# BBox type: (x1, y1, x2, y2, conf)
BBox = Tuple[int, int, int, int, float]


class FaceEngine:
    """
    Kết hợp YOLOv8-face (detect) và InsightFace ArcFace (embed).
    Sử dụng như singleton qua FaceEngine.instance().
    """

    _instance: Optional["FaceEngine"] = None

    @classmethod
    def instance(cls) -> "FaceEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._yolo = None
        self._app  = None   # InsightFace FaceAnalysis
        self._loaded = False

    def load(self):
        """Tải model"""
        if self._loaded:
            return
        # Bắt buộc load InsightFace (onnxruntime) TRƯỚC YOLO (torch) để tránh lỗi DLL Conflict trên Windows
        self._load_insightface()
        self._load_yolo()
        self._loaded = True
        logger.info("FaceEngine loaded successfully.")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    #  YOLOv8-face    

    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            model_path = MODEL_DIR / YOLO_MODEL_NAME
            self._yolo = YOLO(str(model_path) if model_path.exists() else YOLO_MODEL_NAME)
            logger.info(f"YOLOv8-face loaded: {YOLO_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Cannot load YOLOv8-face: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[BBox]:
        """
        Detect khuôn mặt trong frame.
        Trả về list (x1, y1, x2, y2, conf).
        """
        if self._yolo is None:
            return []
        results = self._yolo.predict(
            frame,
            conf=YOLO_CONF_THRESH,
            iou=YOLO_IOU_THRESH,
            imgsz=YOLO_IMG_SIZE,
            device=YOLO_DEVICE,
            verbose=False,
        )
        boxes: List[BBox] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
        return boxes

    #  InsightFace    

    def _load_insightface(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(
                name=INSIGHTFACE_MODEL,
                root=str(MODEL_DIR),
                providers=["CPUExecutionProvider"] if INSIGHTFACE_DEVICE < 0 else ["CUDAExecutionProvider"],
            )
            self._app.prepare(ctx_id=INSIGHTFACE_DEVICE, det_size=(640, 640))
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
        
        # Mở rộng bbox (padding) khoảng 30% để InsightFace dễ detect ra 5 landmarks
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(bw * 0.3), int(bh * 0.3)
        
        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(w, x2 + pad_x)
        py2 = min(h, y2 + pad_y)
        
        face_crop = frame[py1:py2, px1:px2]
        if face_crop.size == 0:
            return None
            
        # InsightFace nhận ảnh BGR, gọi app.get() trên vùng crop: Detect => Align => Embed
        faces = self._app.get(face_crop)
        if not faces:
            return None
            
        # Nếu có nhiều khuôn mặt trong vùng crop, lấy khuôn mặt to nhất (gần tâm nhất)
        faces = sorted(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
        
        emb = faces[0].normed_embedding   # đã normalize L2
        return emb.astype(np.float32)

    # Các hàm hỗ trợ cho Quản lý dữ liệu (CRUD)
    def get_embedding_from_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Embed từ ảnh full (không cần bbox) — dùng khi thêm người vào DB.
        Trả về None nếu không phát hiện đúng 1 khuôn mặt.
        """
        if self._app is None:
            return None
        faces = self._app.get(image)
        if len(faces) != 1:
            return None
        return faces[0].normed_embedding.astype(np.float32)

    def count_faces(self, image: np.ndarray) -> int:
        """Đếm số khuôn mặt trong ảnh (dùng khi validate CRUD)."""
        if self._app is None:
            return 0
        faces = self._app.get(image)
        return len(faces)
