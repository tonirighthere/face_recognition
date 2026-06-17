"""
Singleton dependencies: FaceEngine, VectorStore, DatabaseManager.
Được khởi tạo một lần khi app startup và tái sử dụng qua FastAPI Depends.
"""
import logging
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE_DIR))

from core.face_engine import FaceEngine
from models.vector_store import VectorStore
from models.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

# Singleton instances (lazy-loaded tại startup)
_db: DatabaseManager | None = None
_engine: FaceEngine | None = None
_store: VectorStore | None = None


def startup():
    """Khởi tạo tất cả singleton – gọi trong lifespan của FastAPI."""
    global _db, _engine, _store
    logger.info("Khởi tạo DatabaseManager...")
    _db = DatabaseManager()

    logger.info("Khởi tạo FaceEngine (MediaPipe + InsightFace)...")
    _engine = FaceEngine.instance()
    _engine.load()

    logger.info("Nạp VectorStore từ DB...")
    _store = VectorStore.instance()
    _store.load_from_db(_db)

    logger.info(f"Hệ thống sẵn sàng — {_store.size} khuôn mặt trong bộ nhớ.")


def get_db() -> DatabaseManager:
    return _db


def get_engine() -> FaceEngine:
    return _engine


def get_store() -> VectorStore:
    return _store
