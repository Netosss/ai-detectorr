FROM python:3.11-slim

# Optimization: Prevent .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install only essential system dependencies
# ffmpeg: required for ffprobe video metadata extraction
# libgl1, libglib2.0-0: required for OpenCV (cv2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Copy only essential application folders and configs
COPY app/ ./app/
COPY configs/ ./configs/

# Ensure the app package is recognized
RUN touch app/__init__.py

# Railway/Cloud Run typically injects the PORT environment variable
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2 --timeout-keep-alive 5
