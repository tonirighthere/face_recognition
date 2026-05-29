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
