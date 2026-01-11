# Use the stable RunPod PyTorch DEVEL image
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Ensure logs are not buffered and prevent interactive prompts
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# --- CACHE & ENVIRONMENT SETTINGS ---
# HF_HOME: Base for model caching
ENV HF_HOME=/app/cache
ENV PYTHONPATH="/app:/app/third_party/grip/TruFor/TruFor_train_test"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY worker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    rm -rf /root/.cache/pip

# --- BAKE MODELS ---
# HuggingFace models (Model A & B)
COPY worker/builder.py /app/builder.py
RUN python /app/builder.py && rm /app/builder.py

# --- CORE FILES (Surgical Copy to save context) ---
# We ONLY copy the necessary TruFor files to avoid transferring unused third_party data
COPY third_party/grip/TruFor /app/third_party/grip/TruFor

# Copy production config
COPY configs/production_v1.json /app/configs/production_v1.json

# Finally copy the handler code
COPY worker/handler.py /app/handler.py

# Entry point
CMD [ "python", "-u", "/app/handler.py" ]
