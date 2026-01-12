import runpod
import os
import asyncio
import base64
import logging
import io
import time
from PIL import Image
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)

# Module-level cache for RunPod endpoint object
_ENDPOINT_CACHE = None

def get_config():
    return {
        "api_key": os.getenv("RUNPOD_API_KEY"),
        "endpoint_id": os.getenv("RUNPOD_ENDPOINT_ID"),
    }

def get_endpoint():
    """Retrieve or initialize the RunPod endpoint (cached)."""
    global _ENDPOINT_CACHE
    config = get_config()
    if not config["endpoint_id"]:
        return None
    if _ENDPOINT_CACHE is None:
        runpod.api_key = config["api_key"]
        _ENDPOINT_CACHE = runpod.Endpoint(config["endpoint_id"])
    return _ENDPOINT_CACHE

def optimize_image(source: Union[str, Image.Image], max_size: int = 1024) -> tuple:
    try:
        if isinstance(source, str):
            img = Image.open(source)
        else:
            img = source

        orig_w, orig_h = img.size
        if max(orig_w, orig_h) > max_size:
            img.thumbnail((max_size, max_size))
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        
        if isinstance(source, str):
            img.close()

        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return encoded, orig_w, orig_h
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return "", 0, 0

async def poll_job(job, timeout=30):
    """Tight async polling loop (200ms sweet spot)."""
    start = time.monotonic()
    # First poll very quickly
    await asyncio.sleep(0.1)
    
    while True:
        status_raw = job.status()
        
        # Robust status check (handles both string and dict responses)
        if isinstance(status_raw, dict):
            status = status_raw.get("status")
        else:
            status = status_raw
        
        if status == "COMPLETED":
            # Robust output retrieval (handles embedded output in status)
            if isinstance(status_raw, dict) and "output" in status_raw:
                return status_raw["output"]
            return job.output()

        if status in ("FAILED", "CANCELLED"):
            error_details = status_raw if isinstance(status_raw, dict) else status
            raise RuntimeError(f"RunPod job {status}: {error_details}")

        if time.monotonic() - start > timeout:
            raise TimeoutError(f"Inference timed out after {timeout}s")

        await asyncio.sleep(0.2)

async def run_deep_forensics(source: Union[str, Image.Image], width: int = 0, height: int = 0) -> Dict[str, Any]:
    endpoint = get_endpoint()
    if not endpoint:
        return {"output": {}, "gpu_time_ms": 0.0, "error": "No endpoint ID configured"}

    try:
        image_base64, w, h = optimize_image(source, max_size=1024)
        
        payload = {
            "image": image_base64,
            "original_width": width if width > 0 else w,
            "original_height": height if height > 0 else h,
            "task": "deep_forensic"
        }

        logger.info(f"[RUNPOD] Starting tight polling for job...")
        job = endpoint.run(payload)
        job_result = await poll_job(job)
        
        # Extract metadata from the new ensemble worker schema
        output = job_result.get("results", {}) if isinstance(job_result, dict) else {}
        if not output and isinstance(job_result, dict):
            output = job_result # Fallback for flat structure
            
        timing = job_result.get("timing_ms", 0.0)
        gpu_time_ms = timing.get("total", 0.0) if isinstance(timing, dict) else float(timing)

        return {
            "output": output,
            "gpu_time_ms": gpu_time_ms,
            "error": None
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[RUNPOD] Polling failed: {error_msg}")
        return {"output": {}, "gpu_time_ms": 0.0, "error": error_msg}

async def run_batch_forensics(frames: list) -> Dict[str, Any]:
    """Tight polling for video frames."""
    endpoint = get_endpoint()
    if not endpoint or not frames:
        return {"results": [], "gpu_time_ms": 0.0}

    try:
        # Batch encode
        images_b64 = []
        for frame in frames:
            img = frame.copy()
            if max(img.size) > 512: img.thumbnail((512, 512))
            if img.mode != "RGB": img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            images_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        job = endpoint.run({"images": images_b64, "task": "deep_forensic"})
        job_result = await poll_job(job)
        
        results = job_result.get("results", []) if isinstance(job_result, dict) else []
        timing = job_result.get("timing_ms", 0.0)
        gpu_time_ms = timing.get("total", 0.0) if isinstance(timing, dict) else float(timing)

        return {"results": results, "gpu_time_ms": gpu_time_ms, "error": None}
    except Exception as e:
        return {"results": [], "gpu_time_ms": 0.0, "error": str(e)}

async def run_image_removal(image_path: str) -> Dict[str, Any]:
    endpoint = get_endpoint()
    if not endpoint: return {"error": "No endpoint"}
    image_base64, _, _ = optimize_image(image_path)
    job = endpoint.run({"image": image_base64, "task": "image_removal"})
    return await poll_job(job)

async def run_video_removal(video_path: str) -> Dict[str, Any]:
    endpoint = get_endpoint()
    if not endpoint: return {"error": "No endpoint"}
    with open(video_path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode("utf-8")
    job = endpoint.run({"video": video_base64, "task": "video_removal"})
    return await poll_job(job)
