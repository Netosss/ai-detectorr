# Use the stable RunPod PyTorch DEVEL image
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Ensure logs are not buffered and prevent interactive prompts
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# --- CACHE & ENVIRONMENT SETTINGS ---
ENV HF_HOME=/app/cache
# We set PYTHONPATH to include common locations, but handler.py also does dynamic injection
ENV PYTHONPATH="/app:/app/third_party/grip/TruFor/TruFor_train_test:${PYTHONPATH}"

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
COPY worker/builder.py /app/builder.py
RUN python /app/builder.py && rm /app/builder.py

# --- CORE FILES ---
# Copy the entire third_party folder to maintain complex submodule structures
COPY third_party /app/third_party

# Copy production config
COPY configs/production_v1.json /app/configs/production_v1.json

# Finally copy the handler code
COPY worker/handler.py /app/handler.py

# Entry point
CMD [ "python", "-u", "/app/handler.py" ]
