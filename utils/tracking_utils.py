from typing import Tuple

BBox = Tuple[int, int, int, int, float]

def calculate_iou(a: BBox, b: BBox) -> float:
    """
    Tính IoU giữa 2 bounding boxes.
    BBox format: (x1, y1, x2, y2, conf)
    """
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

def should_re_embed(
    track, 
    current_area: float, 
    current_conf: float, 
    current_time: float, 
    cooldown: float = 1.0, 
    delta_conf: float = 0.1,
    area_ratio: float = 1.2
) -> bool:
    """
    Quyết định xem có cần gửi lại khuôn mặt này cho InsightFace để lấy embedding mới không.
    Điều kiện:
      - Chưa được nhận diện.
      - HOẶC (đã qua thời gian cooldown VÀ (diện tích mặt to hơn 20% HOẶC độ nét cao hơn delta_conf)).
    """
    if not track.recognized:
        return True
    
    better_quality = (current_area > track.best_area * area_ratio) or (current_conf > track.best_conf + delta_conf)
    time_passed = (current_time - track.last_embed_time) > cooldown
    
    return better_quality and time_passed

def update_embedding_ema(old_emb, new_emb, alpha: float = 0.8):
    """
    Cập nhật vector embedding theo Exponential Moving Average (EMA).
    Giúp vector mượt mà hơn qua các frame.
    Trả về vector đã được normalize (chuẩn hóa L2).
    """
    import numpy as np
    updated_emb = alpha * old_emb + (1.0 - alpha) * new_emb
    return updated_emb / np.linalg.norm(updated_emb)
