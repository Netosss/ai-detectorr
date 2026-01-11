# Use the stable RunPod PyTorch DEVEL image (runtime is often missing or inconsistent)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Ensure logs are not buffered and prevent interactive prompts
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# --- CACHE SETTINGS ---
# Bake models into the image to eliminate runtime downloads and reduce cold start
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache

WORKDIR /app

# Install system dependencies if needed (TruFor might need some)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY worker/requirements.txt .

# Install dependencies and clean up cache in same layer
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# --- BAKE MODELS ---
# Copy builder script and download models (Model A & B)
# This is done before copying the full handler code to keep the layer stable
COPY worker/builder.py .
RUN python builder.py && \
    rm builder.py

# Copy third_party forensic dependencies (TruFor)
# Copying this before the handler code as it changes less frequently
COPY third_party /app/third_party

# Copy production config
COPY configs/production_v1.json /app/configs/production_v1.json

# Finally copy the handler code (this changes most frequently)
COPY worker/handler.py .

# RunPod expects this entry point
CMD [ "python", "-u", "handler.py" ]
