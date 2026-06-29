"""
Router: Nhận diện khuôn mặt
POST /recognize/image  — Nhận diện từ ảnh tĩnh (upload)
WS   /recognize/stream — WebSocket stream realtime (base64 JPEG frames)

Tối ưu WebSocket stream:
  • MediaPipe detect mỗi frame với quality filter (blur / roll / yaw / pitch)
  • Lọc mặt có diện tích < MIN_FACE_AREA trước khi gọi InsightFace
  • SimpleTracker giữ track_id ổn định, cache kết quả nhận diện
  • Chỉ re-embed khi track MỚI hoặc đã quá TRACK_COOLDOWN giây
"""

import base64
import json
import logging
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from PIL import Image, ImageDraw, ImageFont


from api.dependencies import get_db, get_engine, get_store
from config import (
    SIMILARITY_THRESHOLD,
    BBOX_COLOR_KNOWN, BBOX_COLOR_UNKNOWN,
    TRACK_COOLDOWN, TRACK_IOU_THRESHOLD, TRACK_MAX_LOST,
    MIN_FACE_AREA,
    SKIP_FRAME,
    TRACK_DELTA_CONF, TRACK_AREA_RATIO, TRACK_EMA_ALPHA,
)
from utils.tracking_utils import should_re_embed, update_embedding_ema
from core.tracker import SimpleTracker, Track

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recognize", tags=["Recognize"])

# Scale để resize ảnh trước khi detect (giảm tải MediaPipe)
DETECT_SCALE = 0.5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Không thể đọc ảnh.")
    return img


def _draw_unicode_text_on_cv2(
    img: np.ndarray,
    text: str,
    pos: tuple,
    font_size: int = 14,
    color: tuple = (255, 255, 255),
    bg_color: tuple = (0, 229, 100),
) -> np.ndarray:
    """Vẽ text Unicode tiếng Việt có nền màu lên ảnh OpenCV (BGR) sử dụng Pillow."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font = None
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:\\Windows\\Fonts\\Arial.ttf",
        "arial.ttf"
    ]
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except Exception:
            continue

    if font is None:
        font = ImageFont.load_default()

    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        tw = right - left
        th = bottom - top
    except AttributeError:
        tw, th = draw.textsize(text, font=font)

    x, y = pos
    # Vẽ hộp nền chữ phía trên bounding box
    draw.rectangle([x, y - th - 8, x + tw + 8, y], fill=bg_color)
    # Vẽ text
    draw.text((x + 4, y - th - 5), text, font=font, fill=color)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _draw_track(frame: np.ndarray, track: Track) -> None:
    """Vẽ bbox + label lên frame theo trạng thái track (không dùng Unicode, chỉ giữ lại để tương thích nếu cần)."""
    x1, y1, x2, y2, _ = track.bbox
    color = BBOX_COLOR_KNOWN if track.recognized else BBOX_COLOR_UNKNOWN

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if track.recognized and track.person_name:
        label = f"{track.person_name} ({track.similarity:.2f})"
    elif track.embedding is None:
        label = "Analyzing..."
    else:
        label = "Unknown"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame, label, (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
    )


def _process_frame(
    frame_bgr: np.ndarray,
    engine,
    store,
    threshold: float = SIMILARITY_THRESHOLD,
    apply_filter: bool = False,
) -> tuple[np.ndarray, list]:
    """
    Detect + embed + search → vẽ bbox.
    Dùng cho /recognize/image (ảnh tĩnh) — không cần tracker.
    """
    bboxes = engine.detect(frame_bgr, apply_filter=apply_filter)
    results = []

    for bbox in bboxes:
        x1, y1, x2, y2, score = bbox
        emb = engine.get_embedding(frame_bgr, bbox)

        name       = "Unknown"
        sim        = 0.0
        pid        = None
        recognized = False

        if emb is not None:
            res = store.search_best(emb, threshold=threshold)
            if res is not None:
                pid, name, sim = res
                recognized = True

        color = BBOX_COLOR_KNOWN if recognized else BBOX_COLOR_UNKNOWN
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        label = f"{name} ({sim:.2f})" if recognized else "Unknown"
        frame_bgr = _draw_unicode_text_on_cv2(
            frame_bgr, label, (x1, y1),
            font_size=14, color=(255, 255, 255), bg_color=color
        )

        results.append({
            "person_id":  pid,
            "name":       name,
            "similarity": round(float(sim), 4),
            "recognized": recognized,
            "bbox":       [x1, y1, x2, y2],
            "confidence": round(float(score), 4),
        })

    return frame_bgr, results


# ── REST endpoint ─────────────────────────────────────────────────────────────

@router.post("/image", summary="Nhận diện khuôn mặt từ ảnh tĩnh")
async def recognize_image(
    photo:        UploadFile = File(...),
    threshold:    float      = Form(SIMILARITY_THRESHOLD),
    apply_filter: bool       = Form(False),
    engine=Depends(get_engine),
    store=Depends(get_store),
):
    raw   = await photo.read()
    frame = _decode_image(raw)
    result_frame, detections = _process_frame(frame, engine, store, threshold, apply_filter)

    _, buf  = cv2.imencode(".jpg", result_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf).decode()

    return {
        "image_base64": img_b64,
        "detections":   detections,
        "face_count":   len(detections),
    }


# ── WebSocket realtime stream ─────────────────────────────────────────────────

@router.websocket("/stream")
async def recognize_stream(websocket: WebSocket):
    """
    Protocol:
      Client → Server: JSON { "frame": "<base64 JPEG>", "threshold": 0.4 }
      Server → Client: JSON { "image_base64": "...", "detections": [...], "fps": 12.5 }

    Pipeline mỗi frame:
      1. MediaPipe detect + quality filter (blur/roll/yaw/pitch) — apply_filter=True
      2. Lọc bbox có diện tích < MIN_FACE_AREA
      3. SimpleTracker.update() → ghép bbox vào track_id ổn định
      4. Mỗi track chỉ gọi InsightFace + cosine search khi:
           - track.embedding is None  (track mới, chưa từng embed)
           - hoặc (now - track.last_embed_time) >= TRACK_COOLDOWN
      5. Các frame còn lại dùng lại cache (person_name, similarity) từ track
    """
    engine  = get_engine()
    store   = get_store()
    tracker = SimpleTracker(iou_threshold=TRACK_IOU_THRESHOLD, max_lost=TRACK_MAX_LOST)

    await websocket.accept()
    logger.info("WebSocket /stream kết nối mới.")

    t_prev = time.perf_counter()
    fps    = 0.0
    frame_count = 0

    try:
        while True:
            raw_msg = await websocket.receive_text()
            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "JSON không hợp lệ"}))
                continue

            frame_b64 = msg.get("frame", "")
            threshold = float(msg.get("threshold", SIMILARITY_THRESHOLD))

            if not frame_b64:
                continue

            # Decode frame
            try:
                raw_bytes = base64.b64decode(frame_b64)
                frame_bgr = _decode_image(raw_bytes)
            except Exception:
                continue

            frame_count += 1
            is_detect_frame = (SKIP_FRAME <= 1) or (frame_count % SKIP_FRAME == 1)

            if is_detect_frame:
                # ── 1. Resize xuống DETECT_SCALE để MediaPipe chạy nhanh hơn ───────────
                h_orig, w_orig = frame_bgr.shape[:2]
                detect_w = max(1, int(w_orig * DETECT_SCALE))
                detect_h = max(1, int(h_orig * DETECT_SCALE))
                detect_frame = cv2.resize(frame_bgr, (detect_w, detect_h))

                # ── 2. Detect trên ảnh nhỏ, scale bbox về kích thước gốc ─────────
                bboxes_small = engine.detect(detect_frame, apply_filter=True)
                inv = 1.0 / DETECT_SCALE
                bboxes = [
                    (int(x1 * inv), int(y1 * inv),
                     int(x2 * inv), int(y2 * inv), s)
                    for x1, y1, x2, y2, s in bboxes_small
                ]

                # ── 3. Lọc diện tích tối thiểu (theo kích thước gốc) ─────────────────
                bboxes = [
                    b for b in bboxes
                    if (b[2] - b[0]) * (b[3] - b[1]) >= MIN_FACE_AREA
                ]

                # ── 4. Cập nhật tracker ────────────────────────────────────────────
                tracks = tracker.update(bboxes)
            else:
                # Dùng các track active cũ (không chạy detect)
                tracks = tracker.active_tracks

            # ── 4. Nhận diện có điều kiện + vẽ ───────────────────────────────
            now        = time.time()
            detections = []

            for track in tracks:
                # Bỏ qua track đang bị mất (không xuất hiện frame này)
                if track.lost > 0:
                    continue

                x1, y1, x2, y2, score = track.bbox
                current_area = (x2 - x1) * (y2 - y1)

                need_embed = (
                    is_detect_frame and (
                        track.embedding is None                              # chưa từng embed
                        or should_re_embed(
                            track, current_area, score, now,
                            cooldown=TRACK_COOLDOWN,
                            delta_conf=TRACK_DELTA_CONF,
                            area_ratio=TRACK_AREA_RATIO
                        )
                    )
                )

                if need_embed:
                    emb = engine.get_embedding(frame_bgr, track.bbox)
                    if emb is not None:
                        if track.embedding is None:
                            track.embedding = emb
                        else:
                            # Áp dụng EMA smoothing để ổn định vector embedding qua các frame
                            track.embedding = update_embedding_ema(
                                track.embedding, emb, alpha=TRACK_EMA_ALPHA
                            )

                        track.last_embed_time = now

                        # Cập nhật thông số chất lượng tốt nhất của track
                        if current_area > track.best_area:
                            track.best_area = current_area
                        if score > track.best_conf:
                            track.best_conf = score

                        res     = store.search_best(track.embedding, threshold=threshold)
                        new_sim = res[2] if res is not None else 0.0

                        # Chỉ cập nhật nếu kết quả mới tốt hơn, hoặc nhận ra người khác,
                        # hoặc lần đầu tiên nhận diện (track.similarity == 0)
                        should_update = (
                            track.similarity == 0.0            # lần đầu
                            or new_sim > track.similarity      # chính xác hơn
                            or (res is not None                # đổi sang người khác dù thấp hơn không đáng kể
                                and res[0] != track.person_id)
                        )

                        if should_update:
                            if res is not None:
                                track.person_id   = res[0]
                                track.person_name = res[1]
                                track.similarity  = new_sim
                                track.recognized  = True
                            else:
                                track.person_id   = None
                                track.person_name = None
                                track.similarity  = 0.0
                                track.recognized  = False

                # Không cần vẽ bbox lên server frame nữa vì client tự vẽ, tiết kiệm CPU
                pass

                x1, y1, x2, y2, score = track.bbox
                detections.append({
                    "track_id":   track.track_id,
                    "person_id":  track.person_id,
                    "name":       track.person_name or "Unknown",
                    "similarity": round(float(track.similarity), 4),
                    "recognized": track.recognized,
                    "bbox":       [x1, y1, x2, y2],
                    "confidence": round(float(score), 4),
                })

            # ── FPS ─────────────────────────────────────────────────────────────────
            now_perf = time.perf_counter()
            fps      = 1.0 / max(now_perf - t_prev, 1e-6)
            t_prev   = now_perf

            # ── Chỉ gửi JSON detections, KHÔNG encode ảnh về ──────────────────────────────
            # Client tự vẽ bbox lên canvas trực tiếp từ video stream (nhanh hơn rất nhiều)
            await websocket.send_text(json.dumps({
                "detections": detections,
                "fps":        round(fps, 1),
            }))

    except WebSocketDisconnect:
        logger.info("WebSocket /stream ngắt kết nối.")
    except Exception as e:
        logger.error(f"WebSocket /stream lỗi: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
    finally:
        tracker.reset()
        logger.debug("Tracker đã reset sau khi WebSocket đóng.")
