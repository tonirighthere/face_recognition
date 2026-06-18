"""
Face Recognition System — FastAPI Backend
=========================================
Endpoints:
  POST /api/recognize          — Nhận diện khuôn mặt từ frame ảnh (base64 hoặc file)
  POST /api/register           — Đăng ký nhân sự mới
  GET  /api/personnel          — Lấy danh sách nhân sự
  GET  /api/personnel/{id}     — Lấy thông tin một nhân sự
  PUT  /api/personnel/{id}     — Cập nhật thông tin nhân sự
  DELETE /api/personnel/{id}   — Xóa nhân sự
  GET  /api/personnel/search   — Tìm kiếm nhân sự
  GET  /api/health             — Health check
  GET  /docs                   — Swagger UI (auto-generated)
"""

import base64
import io
import logging
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Setup path ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from config import APP_NAME, APP_VERSION, STORAGE_DIR, SIMILARITY_THRESHOLD
from core.face_engine import FaceEngine
from models.vector_store import VectorStore
from models.db_manager import DatabaseManager

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("API")

# ── Init AI & DB (startup) ───────────────────────────────────────────────────
logger.info("Đang khởi tạo các model AI và Database...")
db_manager  = DatabaseManager()
face_engine = FaceEngine.instance()
face_engine.load()
vector_store = VectorStore.instance()
vector_store.load_from_db(db_manager)
logger.info(f"Hệ thống sẵn sàng — {vector_store.size} nhân sự trong DB.")

# ── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title=APP_NAME,
    description="Face Recognition System API — MediaPipe + InsightFace + SQLite",
    version=APP_VERSION,
)

# Cho phép tất cả các origin (cần thiết khi Web UI chạy từ máy khác)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Web UI tĩnh
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class RecognizeRequest(BaseModel):
    """Nhận frame dưới dạng base64 JPEG/PNG để nhận diện realtime."""
    image_base64: str
    apply_filter: bool = True
    threshold: float = SIMILARITY_THRESHOLD


class FaceResult(BaseModel):
    x1: int; y1: int; x2: int; y2: int
    confidence: float
    person_id: Optional[int] = None
    ho_ten: Optional[str] = None
    similarity: Optional[float] = None
    status: str  # "known" | "unknown" | "error"


class RecognizeResponse(BaseModel):
    faces: List[FaceResult]
    face_count: int
    annotated_image_base64: Optional[str] = None


class PersonInfo(BaseModel):
    id: int
    ho_ten: str
    ngay_sinh: Optional[str] = None
    cccd: Optional[str] = None
    gioi_tinh: Optional[str] = None
    dien_thoai: Optional[str] = None
    anh_path: Optional[str] = None
    created_at: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def decode_base64_image(b64_str: str) -> np.ndarray:
    """Giải mã base64 → OpenCV BGR numpy array."""
    # Strip data URI prefix nếu có (data:image/jpeg;base64,...)
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không thể giải mã ảnh từ base64.")
    return img


def encode_image_to_base64(img_bgr: np.ndarray, quality: int = 80) -> str:
    """Encode OpenCV BGR image → base64 JPEG string."""
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def annotate_frame(frame: np.ndarray, faces: List[FaceResult]) -> np.ndarray:
    """Vẽ bounding box + tên lên frame."""
    out = frame.copy()
    for f in faces:
        color = (0, 220, 90) if f.status == "known" else (60, 60, 240)
        cv2.rectangle(out, (f.x1, f.y1), (f.x2, f.y2), color, 3)

        if f.status == "known":
            label = f"{f.ho_ten} ({f.similarity:.2f})"
        elif f.status == "unknown":
            label = "Unknown"
        else:
            label = "Error"

        # Vẽ nền text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(out, (f.x1, f.y1 - 28), (f.x1 + tw + 6, f.y1), color, -1)
        cv2.putText(out, label, (f.x1 + 3, f.y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    """Serve Web UI chính."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": f"{APP_NAME} API đang chạy. Truy cập /docs để xem Swagger UI."})


@app.get("/api/health", tags=["System"])
async def health_check():
    """Kiểm tra trạng thái hệ thống."""
    return {
        "status": "ok",
        "app": APP_NAME,
        "version": APP_VERSION,
        "models_loaded": face_engine.is_loaded,
        "personnel_count": vector_store.size,
    }


# ── Nhận diện khuôn mặt ──────────────────────────────────────────────────────

@app.post("/api/recognize", response_model=RecognizeResponse, tags=["Recognition"])
async def recognize(req: RecognizeRequest):
    """
    Nhận diện khuôn mặt từ frame base64.
    Trả về danh sách khuôn mặt nhận diện được + ảnh đã annotate.
    """
    try:
        frame = decode_base64_image(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ảnh không hợp lệ: {e}")

    bboxes = face_engine.detect(frame, apply_filter=req.apply_filter)

    face_results: List[FaceResult] = []
    for bbox in bboxes:
        x1, y1, x2, y2, conf = bbox
        emb = face_engine.get_embedding(frame, bbox)

        if emb is None:
            face_results.append(FaceResult(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=round(conf, 3),
                status="error",
            ))
            continue

        match = vector_store.search_best(emb, threshold=req.threshold)
        if match:
            pid, ho_ten, sim = match
            face_results.append(FaceResult(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=round(conf, 3),
                person_id=pid,
                ho_ten=ho_ten,
                similarity=round(sim, 4),
                status="known",
            ))
        else:
            face_results.append(FaceResult(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=round(conf, 3),
                status="unknown",
            ))

    # Annotate và encode lại ảnh
    annotated = annotate_frame(frame, face_results)
    annotated_b64 = encode_image_to_base64(annotated)

    return RecognizeResponse(
        faces=face_results,
        face_count=len(face_results),
        annotated_image_base64=annotated_b64,
    )


# ── Đăng ký nhân sự ──────────────────────────────────────────────────────────

@app.post("/api/register", tags=["Personnel"])
async def register_person(
    ho_ten:     str        = Form(..., description="Họ và tên (bắt buộc)"),
    cccd:       str        = Form(..., description="CCCD / Mã nhân viên (bắt buộc)"),
    gioi_tinh:  str        = Form("Nam"),
    ngay_sinh:  str        = Form(""),
    dien_thoai: str        = Form(""),
    image:      UploadFile = File(..., description="Ảnh chân dung (JPEG/PNG)"),
):
    """
    Đăng ký nhân sự mới vào hệ thống.
    Nhận file ảnh multipart/form-data. Trích xuất embedding và lưu vào DB.
    """
    # Đọc và decode ảnh
    raw = await image.read()
    arr = np.frombuffer(raw, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="File ảnh không hợp lệ hoặc không đọc được.")

    # Kiểm tra số khuôn mặt
    face_count = face_engine.count_faces(img_bgr)
    if face_count == 0:
        raise HTTPException(status_code=422, detail="Không tìm thấy khuôn mặt trong ảnh. Hãy dùng ảnh rõ mặt hơn.")
    if face_count > 1:
        raise HTTPException(status_code=422, detail=f"Phát hiện {face_count} khuôn mặt. Chỉ chấp nhận ảnh có đúng 1 khuôn mặt.")

    # Trích xuất embedding
    emb = face_engine.get_embedding_from_image(img_bgr)
    if emb is None:
        raise HTTPException(status_code=422, detail="Không thể trích xuất đặc trưng khuôn mặt. Ảnh có thể quá mờ hoặc không đủ ánh sáng.")

    # Lưu ảnh vào storage
    photo_name = f"{cccd}.jpg"
    photo_path = STORAGE_DIR / photo_name
    cv2.imwrite(str(photo_path), img_bgr)

    # Lưu vào DB
    try:
        person_id = db_manager.add_person(
            ho_ten=ho_ten,
            ngay_sinh=ngay_sinh,
            cccd=cccd,
            gioi_tinh=gioi_tinh,
            dien_thoai=dien_thoai,
            anh_path=str(photo_path),
            embedding=emb,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi cơ sở dữ liệu: {e}")

    # Cập nhật vector store RAM cache
    vector_store.add(person_id, ho_ten, emb)

    return {
        "success": True,
        "person_id": person_id,
        "message": f"Đã đăng ký thành công: {ho_ten} (ID: {person_id})",
    }


# ── Quản lý nhân sự (CRUD) ────────────────────────────────────────────────────

@app.get("/api/personnel", response_model=List[PersonInfo], tags=["Personnel"])
async def list_personnel():
    """Lấy toàn bộ danh sách nhân sự."""
    return db_manager.get_all()


@app.get("/api/personnel/search", response_model=List[PersonInfo], tags=["Personnel"])
async def search_personnel(q: str = Query(..., description="Từ khoá tìm kiếm (tên hoặc CCCD)")):
    """Tìm kiếm nhân sự theo tên hoặc CCCD."""
    return db_manager.search(q)


@app.get("/api/personnel/{person_id}", response_model=PersonInfo, tags=["Personnel"])
async def get_person(person_id: int):
    """Lấy thông tin một nhân sự theo ID."""
    person = db_manager.get_by_id(person_id)
    if not person:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy nhân sự ID={person_id}")
    return person


@app.delete("/api/personnel/{person_id}", tags=["Personnel"])
async def delete_person(person_id: int):
    """Xóa nhân sự khỏi hệ thống (DB + ảnh + vector cache)."""
    person = db_manager.get_by_id(person_id)
    if not person:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy nhân sự ID={person_id}")

    # Xóa ảnh
    if person.get("anh_path"):
        p = Path(person["anh_path"])
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    success = db_manager.delete_person(person_id)
    if not success:
        raise HTTPException(status_code=500, detail="Lỗi khi xóa khỏi cơ sở dữ liệu.")

    # Reload vector store
    vector_store.load_from_db(db_manager)

    return {"success": True, "message": f"Đã xóa nhân sự '{person['ho_ten']}' (ID: {person_id})"}
