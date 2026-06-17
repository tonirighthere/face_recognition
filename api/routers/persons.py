"""
Router: Quản lý nhân sự (CRUD)
POST   /persons/           — Đăng ký nhân sự mới (upload ảnh + thông tin)
GET    /persons/           — Danh sách tất cả nhân sự
GET    /persons/{id}       — Chi tiết 1 nhân sự
DELETE /persons/{id}       — Xóa nhân sự
GET    /persons/search     — Tìm kiếm theo tên / CCCD
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api.dependencies import get_db, get_engine, get_store
from config import STORAGE_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/persons", tags=["Persons"])


# ── Helpers ──────────────────────────────────────────────────────────────────

def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Không thể đọc ảnh — định dạng không hỗ trợ.")
    return img


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/", summary="Đăng ký nhân sự mới")
async def register_person(
    ho_ten:     str       = Form(..., description="Họ và tên"),
    cccd:       str       = Form(..., description="Số CCCD / Mã nhân viên"),
    ngay_sinh:  str       = Form("",  description="Ngày sinh (YYYY-MM-DD)"),
    gioi_tinh:  str       = Form("Nam", description="Giới tính"),
    dien_thoai: str       = Form("",  description="Số điện thoại"),
    photo:      UploadFile = File(..., description="Ảnh chân dung (jpg/png)"),
    db=Depends(get_db),
    engine=Depends(get_engine),
    store=Depends(get_store),
):
    raw = await photo.read()
    img_bgr = _decode_image(raw)

    face_count = engine.count_faces(img_bgr)
    if face_count == 0:
        raise HTTPException(422, "Không phát hiện khuôn mặt nào trong ảnh.")
    if face_count > 1:
        raise HTTPException(422, f"Phát hiện {face_count} khuôn mặt — chỉ chấp nhận ảnh 1 người.")

    emb = engine.get_embedding_from_image(img_bgr)
    if emb is None:
        raise HTTPException(422, "Không thể trích xuất embedding — ảnh quá mờ hoặc chất lượng thấp.")

    photo_path = STORAGE_DIR / f"{cccd}.jpg"
    cv2.imwrite(str(photo_path), img_bgr)

    try:
        person_id = db.add_person(
            ho_ten=ho_ten, ngay_sinh=ngay_sinh, cccd=cccd,
            gioi_tinh=gioi_tinh, dien_thoai=dien_thoai,
            anh_path=str(photo_path), embedding=emb,
        )
        store.add(person_id, ho_ten, emb)
        logger.info(f"Đã đăng ký: {ho_ten} (ID={person_id})")
        return {"success": True, "person_id": person_id, "message": f"Đăng ký thành công: {ho_ten}"}
    except Exception as e:
        raise HTTPException(500, f"Lỗi cơ sở dữ liệu: {e}")


@router.get("/", summary="Danh sách tất cả nhân sự")
def list_persons(db=Depends(get_db)):
    return db.get_all()


@router.get("/search", summary="Tìm kiếm nhân sự")
def search_persons(q: str = "", db=Depends(get_db)):
    if not q:
        return db.get_all()
    return db.search(q)


@router.get("/{person_id}", summary="Chi tiết nhân sự")
def get_person(person_id: int, db=Depends(get_db)):
    person = db.get_by_id(person_id)
    if not person:
        raise HTTPException(404, f"Không tìm thấy nhân sự ID={person_id}")
    # Không trả về blob embedding
    person.pop("embedding", None)
    return person


@router.get("/{person_id}/photo", summary="Ảnh chân dung nhân sự")
def get_person_photo(person_id: int, db=Depends(get_db)):
    person = db.get_by_id(person_id)
    if not person:
        raise HTTPException(404, "Không tìm thấy nhân sự.")
    path = person.get("anh_path", "")
    if not path or not Path(path).exists():
        raise HTTPException(404, "Ảnh chân dung chưa được tải lên.")
    return FileResponse(path, media_type="image/jpeg")


@router.delete("/{person_id}", summary="Xóa nhân sự")
def delete_person(person_id: int, db=Depends(get_db), store=Depends(get_store)):
    person = db.get_by_id(person_id)
    if not person:
        raise HTTPException(404, f"Không tìm thấy nhân sự ID={person_id}")

    anh_path = person.get("anh_path", "")
    if anh_path and Path(anh_path).exists():
        try:
            Path(anh_path).unlink()
        except Exception as e:
            logger.warning(f"Không xóa được file ảnh: {e}")

    success = db.delete_person(person_id)
    if not success:
        raise HTTPException(500, "Lỗi xóa khỏi database.")

    # Reload lại VectorStore
    store.load_from_db(db)
    return {"success": True, "message": f"Đã xóa nhân sự '{person['ho_ten']}' (ID={person_id})"}
