# ============================================================
# Face Recognition System — Docker Image
# FastAPI + InsightFace + MediaPipe  (CPU-only)
# ============================================================
FROM python:3.10-slim-bullseye

LABEL maintainer="atin@company.com"
LABEL description="Face Recognition API — FastAPI + InsightFace + MediaPipe"

# ── System dependencies ──────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        # OpenCV headless runtime
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgl1 \
        libgomp1 \
        # MediaPipe OpenGL/EGL dependencies
        libgles2 \
        libegl1 \
        # Build tools (needed by some pip packages)
        build-essential \
        cmake \
        # Utilities
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (layer cache) ───────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt

# ── Application source ───────────────────────────────────────
COPY . .

# ── Persistent data directories ─────────────────────────────
RUN mkdir -p storage/photos database weights

# ── Environment ──────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV API_PORT=8000

# ── Port ─────────────────────────────────────────────────────
EXPOSE 8000

# ── Startup ──────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
