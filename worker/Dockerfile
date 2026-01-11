# Use the stable RunPod PyTorch DEVEL image
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Ensure logs are not buffered and prevent interactive prompts
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# --- CACHE & ENVIRONMENT SETTINGS ---
# Bake models into the image to eliminate runtime downloads
ENV HF_HOME=/app/cache
ENV PYTHONPATH="/app:/app/third_party/grip/TruFor/TruFor_train_test"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
# Assumes build context is repo root
COPY worker/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# --- BAKE MODELS ---
COPY worker/builder.py .
RUN python builder.py && rm builder.py

# Copy third_party forensic dependencies
COPY third_party ./third_party

# Copy production config
COPY configs/production_v1.json ./configs/production_v1.json

# Finally copy the handler code
COPY worker/handler.py .

# Entry point
CMD [ "python", "-u", "handler.py" ]
