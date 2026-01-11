# Use the stable RunPod PyTorch RUNTIME image (much smaller than devel)
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-runtime-ubuntu22.04

# Ensure logs are not buffered
ENV PYTHONUNBUFFERED=1

# --- CACHE SETTINGS ---
# Bake models into the image to eliminate runtime downloads and reduce cold start
ENV HF_HOME=/app/cache
ENV TRANSFORMERS_CACHE=/app/cache

WORKDIR /app

# Copy requirements first for better layer caching
COPY worker/requirements.txt .

# Install dependencies and clean up cache in same layer
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# --- BAKE MODELS ---
# Copy builder script and download models (Model A & B)
COPY worker/builder.py .
RUN python builder.py && \
    rm builder.py

# Copy the handler code
COPY worker/handler.py .

# Copy third_party forensic dependencies (TruFor)
# NOTE: Build context must be the project root for this to work
COPY third_party /app/third_party

# Copy production config (needed for slice names/info if used in worker, 
# though usually handler has its own logic or hardcoded params)
COPY configs/production_v1.json /app/configs/production_v1.json

# RunPod expects this entry point
CMD [ "python", "-u", "handler.py" ]
