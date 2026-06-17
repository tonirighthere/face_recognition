"""
Router: Nhận diện khuôn mặt
POST /recognize/image   — Nhận diện từ ảnh tĩnh (upload)
WS   /recognize/stream  — WebSocket stream realtime (base64 JPEG frames)
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
from config import SIMILARITY_THRESHOLD, BBOX_COLOR_KNOWN, BBOX_COLOR_UNKNOWN

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/recognize", tags=["Recognize"])


# ── Helpers ──────────────────────────────────────────────────────────────────

def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Không thể đọc ảnh.")
    return img


def _process_frame(
    frame_bgr: np.ndarray,
    engine,
    store,
    threshold: float = SIMILARITY_THRESHOLD,
    apply_filter: bool = False,
) -> tuple[np.ndarray, list]:
    """Detect + embed + search → vẽ bbox → trả về ảnh kết quả và danh sách nhận diện."""
    bboxes = engine.detect(frame_bgr, apply_filter=apply_filter)
    results = []

    for bbox in bboxes:
        x1, y1, x2, y2, score = bbox
        emb = engine.get_embedding(frame_bgr, bbox)

        name = "Unknown"
        sim  = 0.0
        pid  = None
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
        cv2.putText(frame_bgr, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

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
    raw     = await photo.read()
    frame   = _decode_image(raw)
    result_frame, detections = _process_frame(frame, engine, store, threshold, apply_filter)

    # Encode ảnh kết quả về base64 JPEG
    _, buf = cv2.imencode(".jpg", result_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
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
      Client → Server: JSON { "frame": "<base64 JPEG>", "threshold": 0.4, "apply_filter": false }
      Server → Client: JSON { "image_base64": "...", "detections": [...], "fps": 12.5 }
    """
    engine = get_engine()
    store  = get_store()

    await websocket.accept()
    logger.info("WebSocket kết nối mới.")

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

            frame_b64    = msg.get("frame", "")
            threshold    = float(msg.get("threshold", SIMILARITY_THRESHOLD))
            apply_filter = bool(msg.get("apply_filter", False))

            if not frame_b64:
                continue

            # Decode base64 JPEG → numpy BGR
            try:
                raw_bytes = base64.b64decode(frame_b64)
                arr       = np.frombuffer(raw_bytes, np.uint8)
                frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame_bgr is None:
                    continue
            except Exception:
                continue

            result_frame, detections = _process_frame(
                frame_bgr, engine, store, threshold, apply_filter
            )

            # FPS
            now   = time.perf_counter()
            fps   = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now

            # Encode kết quả
            _, buf  = cv2.imencode(".jpg", result_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            img_b64 = base64.b64encode(buf).decode()

            await websocket.send_text(json.dumps({
                "image_base64": img_b64,
                "detections":   detections,
                "fps":          round(fps, 1),
            }))

    except WebSocketDisconnect:
        logger.info("WebSocket ngắt kết nối.")
    except Exception as e:
        logger.error(f"WebSocket lỗi: {e}")
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
