import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def get_pil_font(font_path: str = "arial.ttf", size: int = 16):
    """
    Tải font cho PIL, fallback về font mặc định nếu lỗi.
    """
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()

def draw_text_pil(
    frame: np.ndarray,
    tracks_info: list,
    fps: float,
    font_main,
    font_fps,
    color_known,
    color_unknown,
) -> np.ndarray:
    """
    Vẽ chữ tiếng Việt và bounding box lên frame bằng PIL.
    """
    # Vẽ bbox bằng OpenCV
    for t in tracks_info:
        x1, y1, x2, y2, _ = t["bbox"]
        color = color_known if t["recognized"] else color_unknown
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Chuyển sang PIL để vẽ text tiếng Việt
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img   = Image.fromarray(frame_rgb)
    draw      = ImageDraw.Draw(pil_img)

    for t in tracks_info:
        x1, y1, x2, y2, _ = t["bbox"]
        color     = color_known if t["recognized"] else color_unknown
        color_rgb = (color[2], color[1], color[0])  # BGR → RGB

        if t["recognized"] and t.get("person_name"):
            label = f"ID {t.get('person_id', 0):03d} · {t['person_name']} · {t.get('similarity', 0):.2f}"
        else:
            label = f"Không rõ · #{t.get('track_id', 0):03d}"

        bb = draw.textbbox((0, 0), label, font=font_main)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        lx = int(x1)
        ly = int(max(y1 - 4, th + 8))
        draw.rectangle([(lx, ly - th - 6), (lx + tw + 8, ly + 2)], fill=color_rgb)
        draw.text((lx + 4, ly - th - 4), label, font=font_main, fill=(10, 10, 10))

    # FPS overlay
    draw.text((10, 10), f"FPS: {int(fps)}", font=font_fps, fill=(0, 230, 100))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
