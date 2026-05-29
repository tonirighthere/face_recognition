import math
import logging
import cv2
import numpy as np
from typing import List, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FILTER_BLUR_THRESHOLD,
    FILTER_ROLL_THRESHOLD,
    FILTER_YAW_MIN, FILTER_YAW_MAX,
    FILTER_PITCH_THRESHOLD,
)

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int, float]

def is_face_valid(
    frame: np.ndarray,
    bbox: BBox,
    keypoints_px: List[Tuple[int, int]],
) -> bool:
    """
    Trả về True nếu khuôn mặt đủ chất lượng để embedding.

    Kiểm tra theo thứ tự (nhanh → chậm):
      1. Blur  — Laplacian variance trên vùng crop
      2. Roll  — góc nghiêng đầu qua đường nối 2 mắt
      3. Yaw   — tỷ lệ khoảng cách mũi–2mắt (đã normalize)
      4. Pitch — độ lệch mũi so với đường mắt (đã normalize theo eye_dist)
    """
    x1, y1, x2, y2, _ = bbox

    # --- 1. Blur check --------------------------------------------------
    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return False
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < FILTER_BLUR_THRESHOLD:
        logger.debug(f"Blur reject: {blur_score:.1f} < {FILTER_BLUR_THRESHOLD}")
        return False

    # Cần ít nhất 3 keypoints
    # Tasks API: kp[0]=left_eye, kp[1]=right_eye, kp[2]=nose_tip
    if len(keypoints_px) < 3:
        return True   # Không đủ điểm → bỏ qua kiểm tra pose

    left_eye  = np.array(keypoints_px[0], dtype=np.float32)
    right_eye = np.array(keypoints_px[1], dtype=np.float32)
    nose_tip  = np.array(keypoints_px[2], dtype=np.float32)

    # Khoảng cách 2 mắt — dùng để normalize
    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    if eye_dist < 1e-5:
        return True   # Tránh chia 0

    # --- 2. Roll check --------------------------------------------------
    # θ = arctan((yr - yl) / (xr - xl))
    # right_eye.x > left_eye.x khi mặt thẳng → delta_x > 0 → roll ≈ 0°
    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    roll_deg = math.degrees(math.atan2(delta_y, delta_x))
    if abs(roll_deg) > FILTER_ROLL_THRESHOLD:
        logger.debug(f"Roll reject: {roll_deg:.1f}° > ±{FILTER_ROLL_THRESHOLD}°")
        return False

    # --- 3. Yaw check ---------------------------------------------------
    # r = dist(nose, left_eye) / dist(nose, right_eye)
    dist_nose_left  = float(np.linalg.norm(nose_tip - left_eye))
    dist_nose_right = float(np.linalg.norm(nose_tip - right_eye))
    if dist_nose_right < 1e-5:
        return True
    yaw_ratio = dist_nose_left / dist_nose_right
    if not (FILTER_YAW_MIN <= yaw_ratio <= FILTER_YAW_MAX):
        logger.debug(f"Yaw reject: ratio={yaw_ratio:.2f} not in [{FILTER_YAW_MIN}, {FILTER_YAW_MAX}]")
        return False

    # 4. Pitch check 
    # d = y_nose - (y_left + y_right) / 2  → normalize theo eye_dist
    eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
    d_norm = (nose_tip[1] - eye_mid_y) / eye_dist
    if abs(d_norm) > FILTER_PITCH_THRESHOLD:
        logger.debug(f"Pitch reject: d_norm={d_norm:.2f} > ±{FILTER_PITCH_THRESHOLD}")
        return False

    return True
