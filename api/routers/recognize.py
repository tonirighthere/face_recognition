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

from api.dependencies import get_db, get_engine, get_store
from config import (
    SIMILARITY_THRESHOLD,
    BBOX_COLOR_KNOWN, BBOX_COLOR_UNKNOWN,
    TRACK_COOLDOWN, TRACK_IOU_THRESHOLD, TRACK_MAX_LOST,
    MIN_FACE_AREA,
)
from core.tracker import SimpleTracker, Track

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recognize", tags=["Recognize"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Không thể đọc ảnh.")
    return img


def _draw_track(frame: np.ndarray, track: Track) -> None:
    """Vẽ bbox + label lên frame theo trạng thái track."""
    x1, y1, x2, y2, _ = track.bbox
    color = BBOX_COLOR_KNOWN if track.recognized else BBOX_COLOR_UNKNOWN

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if track.recognized and track.person_name:
        label = f"{track.person_name} ({track.similarity:.2f})"
    elif track.embedding is None:
        label = "Analyzing..."   # chưa kịp embed lần đầu
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
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame_bgr, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
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
                arr       = np.frombuffer(raw_bytes, np.uint8)
                frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue
            except Exception:
                continue

            # ── 1. Detect với quality filter bật sẵn ──────────────────────────
            bboxes = engine.detect(frame_bgr, apply_filter=True)

            # ── 2. Lọc diện tích tối thiểu ─────────────────────────────────────
            bboxes = [
                b for b in bboxes
                if (b[2] - b[0]) * (b[3] - b[1]) >= MIN_FACE_AREA
            ]

            # ── 3. Cập nhật tracker ────────────────────────────────────────────
            tracks = tracker.update(bboxes)

            # ── 4. Nhận diện có điều kiện + vẽ ───────────────────────────────
            now        = time.time()
            detections = []

            for track in tracks:
                # Bỏ qua track đang bị mất (không xuất hiện frame này)
                if track.lost > 0:
                    continue

                need_embed = (
                    track.embedding is None                              # chưa từng embed
                    or (now - track.last_embed_time) >= TRACK_COOLDOWN  # hết cooldown
                )

                if need_embed:
                    emb = engine.get_embedding(frame_bgr, track.bbox)
                    if emb is not None:
                        track.embedding       = emb
                        track.last_embed_time = now

                        res = store.search_best(emb, threshold=threshold)
                        if res is not None:
                            track.person_id   = res[0]
                            track.person_name = res[1]
                            track.similarity  = res[2]
                            track.recognized  = True
                        else:
                            track.person_id   = None
                            track.person_name = None
                            track.similarity  = 0.0
                            track.recognized  = False

                # Vẽ lên frame (dùng cache nếu không re-embed)
                _draw_track(frame_bgr, track)

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

            # ── FPS ────────────────────────────────────────────────────────────
            now_perf = time.perf_counter()
            fps      = 1.0 / max(now_perf - t_prev, 1e-6)
            t_prev   = now_perf

            # ── Encode và gửi ──────────────────────────────────────────────────
            _, buf  = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            img_b64 = base64.b64encode(buf).decode()

            await websocket.send_text(json.dumps({
                "image_base64": img_b64,
                "detections":   detections,
                "fps":          round(fps, 1),
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
