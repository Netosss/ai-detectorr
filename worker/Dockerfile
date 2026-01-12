# Use the stable RunPod PyTorch DEVEL image
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Ensure logs are not buffered and prevent interactive prompts
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# --- CACHE & ENVIRONMENT SETTINGS ---
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache
ENV TORCH_HOME=/app/cache
ENV PYTHONPATH="/app"

WORKDIR /app

# Install system dependencies (Minimal for these models + Pillow-SIMD speed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libjpeg-turbo8-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY worker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip uninstall -y Pillow && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    rm -rf /root/.cache/pip

# --- BAKE MODELS ---
COPY worker/builder.py /app/builder.py
RUN python /app/builder.py && rm /app/builder.py

# --- PRE-WARM CUDA ---
COPY worker/prewarm_cuda.py /app/prewarm_cuda.py
RUN python /app/prewarm_cuda.py && rm /app/prewarm_cuda.py

# --- CONFIG & HANDLER ---
# Copy production config (the winning ensemble config)
COPY configs/new_optimized_config.json /app/configs/production_v1.json

# Finally copy the handler code
COPY worker/handler.py /app/handler.py

# Entry point
CMD [ "python", "-u", "/app/handler.py" ]
