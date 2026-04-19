# syntax=docker/dockerfile:1.6

# ----------------------------------------------------------------------
# Stage 1 — build the React frontend
# ----------------------------------------------------------------------
FROM node:20-alpine AS frontend

WORKDIR /web_ui
COPY web_ui/package.json web_ui/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY web_ui/ ./
RUN npm run build

# ----------------------------------------------------------------------
# Stage 2 — Python runtime
# ----------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenCV + MediaPipe system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the Python server + config + model + phase 2 assets
COPY web_server.py ./
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY phase2/ ./phase2/

# Copy the frontend bundle built in stage 1
COPY --from=frontend /web_ui/dist ./web_ui/dist

ENV HOST=0.0.0.0 \
    PORT=5000
EXPOSE 5000

CMD ["python", "web_server.py"]
