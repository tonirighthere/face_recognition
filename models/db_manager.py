"""
SQLite CRUD operations cho bảng nhan_su
"""

import sqlite3
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any

from config import DB_PATH


class DatabaseManager:
    """Quản lý kết nối và thao tác với SQLite database."""

    _instance: Optional["DatabaseManager"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._db_path = str(DB_PATH)
        self._init_db()
        self._initialized = True

    #  Private    

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Tạo bảng nếu chưa tồn tại."""
        schema_path = Path(__file__).parent / "schema.sql"
        with self._get_conn() as conn:
            if schema_path.exists():
                conn.executescript(schema_path.read_text(encoding="utf-8"))
            else:
                # Fallback inline schema
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS nhan_su (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ho_ten TEXT NOT NULL,
                        ngay_sinh TEXT,
                        cccd TEXT UNIQUE,
                        gioi_tinh TEXT,
                        dien_thoai TEXT,
                        anh_path TEXT,
                        embedding BLOB,
                        created_at TEXT DEFAULT (datetime('now','localtime')),
                        updated_at TEXT DEFAULT (datetime('now','localtime'))
                    );
                """)

    # Encode / Decode embedding 
    @staticmethod
    def encode_embedding(vec: np.ndarray) -> bytes:
        return vec.astype(np.float32).tobytes()

    @staticmethod
    def decode_embedding(blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    # CREATE
    def add_person(
        self,
        ho_ten: str,
        ngay_sinh: str = "",
        cccd: str = "",
        gioi_tinh: str = "Nam",
        dien_thoai: str = "",
        anh_path: str = "",
        embedding: Optional[np.ndarray] = None,
    ) -> int:
        """Thêm người mới. Trả về ID vừa tạo."""
        emb_blob = self.encode_embedding(embedding) if embedding is not None else None
        sql = """
            INSERT INTO nhan_su (ho_ten, ngay_sinh, cccd, gioi_tinh, dien_thoai, anh_path, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._get_conn() as conn:
            cur = conn.execute(sql, (ho_ten, ngay_sinh, cccd, gioi_tinh, dien_thoai, anh_path, emb_blob))
            return cur.lastrowid

    # READ
    def get_all(self) -> List[Dict[str, Any]]:
        """Lấy toàn bộ danh sách (không kèm embedding blob để nhẹ)."""
        sql = "SELECT id, ho_ten, ngay_sinh, cccd, gioi_tinh, dien_thoai, anh_path, created_at FROM nhan_su ORDER BY ho_ten COLLATE NOCASE ASC"
        with self._get_conn() as conn:
            rows = conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def get_by_id(self, person_id: int) -> Optional[Dict[str, Any]]:
        sql = "SELECT * FROM nhan_su WHERE id = ?"
        with self._get_conn() as conn:
            row = conn.execute(sql, (person_id,)).fetchone()
        return dict(row) if row else None

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Lấy id + ho_ten + embedding cho vector search."""
        sql = "SELECT id, ho_ten, embedding FROM nhan_su WHERE embedding IS NOT NULL"
        with self._get_conn() as conn:
            rows = conn.execute(sql).fetchall()
        result = []
        for r in rows:
            if r["embedding"]:
                result.append({
                    "id":     r["id"],
                    "ho_ten": r["ho_ten"],
                    "vec":    self.decode_embedding(r["embedding"]),
                })
        return result

    def search(self, keyword: str) -> List[Dict[str, Any]]:
        """Tìm kiếm theo tên hoặc CCCD."""
        kw = f"%{keyword}%"
        sql = """
            SELECT id, ho_ten, ngay_sinh, cccd, gioi_tinh, dien_thoai, anh_path, created_at
            FROM nhan_su WHERE ho_ten LIKE ? OR cccd LIKE ?
            ORDER BY ho_ten
        """
        with self._get_conn() as conn:
            rows = conn.execute(sql, (kw, kw)).fetchall()
        return [dict(r) for r in rows]

    # UPDATE
    def update_person(
        self,
        person_id: int,
        ho_ten: str,
        ngay_sinh: str = "",
        cccd: str = "",
        gioi_tinh: str = "Nam",
        dien_thoai: str = "",
        anh_path: str = "",
        embedding: Optional[np.ndarray] = None,
    ) -> bool:
        emb_blob = self.encode_embedding(embedding) if embedding is not None else None
        if emb_blob:
            sql = """
                UPDATE nhan_su SET ho_ten=?, ngay_sinh=?, cccd=?, gioi_tinh=?,
                    dien_thoai=?, anh_path=?, embedding=?,
                    updated_at=datetime('now','localtime')
                WHERE id=?
            """
            params = (ho_ten, ngay_sinh, cccd, gioi_tinh, dien_thoai, anh_path, emb_blob, person_id)
        else:
            sql = """
                UPDATE nhan_su SET ho_ten=?, ngay_sinh=?, cccd=?, gioi_tinh=?,
                    dien_thoai=?, anh_path=?,
                    updated_at=datetime('now','localtime')
                WHERE id=?
            """
            params = (ho_ten, ngay_sinh, cccd, gioi_tinh, dien_thoai, anh_path, person_id)
        with self._get_conn() as conn:
            cur = conn.execute(sql, params)
        return cur.rowcount > 0

    # DELETE
    def delete_person(self, person_id: int) -> bool:
        with self._get_conn() as conn:
            cur = conn.execute("DELETE FROM nhan_su WHERE id=?", (person_id,))
        return cur.rowcount > 0

    def count(self) -> int:
        with self._get_conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM nhan_su").fetchone()[0]
