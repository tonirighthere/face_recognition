"""
FastAPI application entry point.
Chạy: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Đưa thư mục gốc vào sys.path
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))

import api.dependencies as deps
from api.routers import persons, recognize
from config import APP_NAME, APP_VERSION

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


# ── Lifespan (startup / shutdown) ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"=== {APP_NAME} v{APP_VERSION} đang khởi động ===")
    deps.startup()
    yield
    logger.info("=== Hệ thống dừng lại ===")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "API nhận diện khuôn mặt realtime sử dụng MediaPipe + InsightFace ArcFace. "
        "Hỗ trợ CRUD nhân sự, nhận diện từ ảnh tĩnh và WebSocket stream."
    ),
    lifespan=lifespan,
)

# ── CORS (cho phép Web UI gọi API) ───────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Routers ───────────────────────────────────────────────────────────────
app.include_router(persons.router, prefix="/api")
app.include_router(recognize.router, prefix="/api")

# ── Static files (Web UI) ─────────────────────────────────────────────────────
STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def serve_ui():
        return FileResponse(str(STATIC_DIR / "index.html"))


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    store = deps.get_store()
    return {
        "status": "ok",
        "app": APP_NAME,
        "version": APP_VERSION,
        "persons_in_memory": store.size if store else 0,
    }
