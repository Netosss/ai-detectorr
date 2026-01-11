import logging
import time
import cv2
import numpy as np
import asyncio
import os
import json
import random
import math
from collections import OrderedDict 
from PIL import Image
from PIL.ExifTags import TAGS
from typing import Optional, Union
from pathlib import Path
from app.c2pa_reader import get_c2pa_manifest
from app.runpod_client import run_deep_forensics
from app.security import security_manager

logger = logging.getLogger(__name__)

# --- PRODUCTION CONFIG LOAD ---
REPO_ROOT = Path(__file__).resolve().parents[2]
PROD_CONFIG_PATH = REPO_ROOT / "configs/production_v1.json"

with open(PROD_CONFIG_PATH, 'r') as f:
    PROD_CONFIG = json.load(f)

# LRU Cache implementation for forensic results
class LRUCache:
    def __init__(self, capacity: int = 1000):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

forensic_cache = LRUCache(capacity=1000)

def get_image_hash(source: Union[str, Image.Image], fast_mode: bool = False) -> str:
    """Generate a secure SHA-256 hash of the image source."""
    if isinstance(source, str):
        with open(source, 'rb') as f:
            return security_manager.get_safe_hash(f.read(2048 * 1024))
    else:
        thumb = source.copy()
        size = (32, 32) if fast_mode else (64, 64)
        thumb.thumbnail(size)
        thumb = thumb.convert("L")
        return security_manager.get_safe_hash(np.array(thumb).tobytes())

def get_exif_data(file_path: str) -> dict:
    """Extract EXIF metadata from the image."""
    try:
        with Image.open(file_path) as img:
            exif = img._getexif() or {}
            exif_data = {}
            for tag, value in exif.items():
                decoded = TAGS.get(tag, tag)
                exif_data[decoded] = value
            return exif_data
    except Exception:
        return {}

def logit(p):
    """Run 30 Logit Transform."""
    p = max(min(p, 0.999999), 0.000001)
    return math.log(p / (1.0 - p))

def get_slice_name(pixels, w, h, ext, aspect):
    """Run 30 Slice Logic."""
    if pixels < 2000: return "thumbnail"
    if 2000 <= pixels < 10000: return "low_res"
    if 10000 <= pixels < 50000: return "10k-50k"
    if 50000 <= pixels < 500000: return "50k-500k"
    if pixels >= 500000: return ">500k"
    if aspect < 0.8: return "portrait_tall"
    if 0.8 <= aspect <= 1.2: return "squareish"
    if ext == ".png": return "png"
    return "default"

def is_frame_quality_ok(frame: np.ndarray, min_brightness: float = 20, min_sharpness: float = 50) -> tuple:
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < min_brightness:
            return False, brightness, 0.0
        h, w = gray.shape
        center_crop = gray[h//4:3*h//4, w//4:3*w//4]
        laplacian_var = cv2.Laplacian(center_crop, cv2.CV_64F).var()
        if laplacian_var < min_sharpness:
            return False, brightness, laplacian_var
        return True, brightness, laplacian_var
    except:
        return True, 128.0, 100.0

def extract_video_frames(video_path: str) -> tuple:
    frames = []
    quality_rejected = 0
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return [], 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: return [], 0
        sample_points = [int(total_frames * 0.20), int(total_frames * 0.50), int(total_frames * 0.80)]
        for pos in sample_points:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret:
                is_ok, brightness, sharpness = is_frame_quality_ok(frame)
                if not is_ok:
                    quality_rejected += 1
                    continue
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        cap.release()
        if quality_rejected > 0:
            logger.info(f"[VIDEO] Skipped {quality_rejected} low-quality frames")
        if len(frames) < 1 and total_frames >= 1:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * 0.5))
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
            cap.release()
    except Exception as e:
        logger.error(f"Error extracting video frames: {e}")
    return frames, quality_rejected

async def get_video_metadata(video_path: str) -> dict:
    try:
        proc = await asyncio.create_subprocess_exec(
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode == 0: return json.loads(stdout.decode())
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
    except Exception as e:
        logger.error(f"Error extracting video metadata: {e}")
    return {}

def get_video_metadata_score(metadata: dict, filename: str = "", file_path: str = "") -> tuple:
    human_score = 0.0
    ai_score = 0.0
    signals = []
    if not metadata: return 0.0, 0.0, ["No metadata"], None
    format_info = metadata.get("format", {})
    tags = format_info.get("tags", {})
    streams = metadata.get("streams", [])
    tags_lower = {k.lower(): v for k, v in tags.items()}
    encoder = str(tags_lower.get("encoder", "")).lower()
    has_android_marker = any(m in k.lower() for m in ["com.android.version", "com.android.capture", "com.samsung"] for k in tags_lower)
    has_ios_marker = any(m in tags_lower for m in ["com.apple.quicktime.make", "com.apple.quicktime.model", "com.apple.quicktime.software"])
    device_marker = has_android_marker or has_ios_marker
    has_ffmpeg = "lavf" in encoder
    has_x264 = "x264" in encoder
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = float(format_info.get("duration", 0))
    fps = 0.0
    if video_stream:
        try:
            fr = video_stream.get("avg_frame_rate", "0/1")
            if "/" in fr:
                num, den = fr.split("/")
                fps = float(num)/float(den) if float(den)>0 else 0
        except: pass
    is_exact_fps = abs(fps-30)<0.001 or abs(fps-60)<0.001 or abs(fps-24)<0.001
    if device_marker: human_score += 0.50
    if fps > 0 and not is_exact_fps: human_score += 0.10
    if any(b in encoder for b in ["iphone", "samsung", "sony", "canon"]): human_score += 0.20
    if has_ffmpeg and has_x264 and not device_marker: ai_score += 0.50
    if is_exact_fps: ai_score += 0.15
    human_score = min(1.0, human_score)
    ai_score = min(1.0, ai_score)
    early_exit = "human" if human_score >= 0.6 and ai_score < 0.3 else "ai" if ai_score >= 0.7 and human_score < 0.2 else None
    return human_score, ai_score, signals, early_exit

def get_forensic_metadata_score(exif: dict) -> tuple:
    score = 0.0
    signals = []
    def to_float(val):
        try: return float(val)
        except: return None
    make = str(exif.get("Make", "")).lower()
    software = str(exif.get("Software", "")).lower()
    if any(m in make for m in ["apple", "google", "samsung", "sony", "canon", "nikon"]):
        score += 0.35
        signals.append("Trusted device manufacturer")
    if any(s in software for s in ["ios", "android", "lightroom"]):
        score += 0.25
        signals.append("Validated vendor pipeline")
    exp = to_float(exif.get("ExposureTime"))
    if exp is not None and 0 < exp < 30:
        score += 0.15
        signals.append("Valid exposure")
    iso = to_float(exif.get("ISOSpeedRatings"))
    if iso is not None and 50 <= iso <= 102400:
        score += 0.15
        signals.append("Realistic ISO")
    f_num = to_float(exif.get("FNumber"))
    if f_num is not None and 0.95 <= f_num <= 32:
        score += 0.15
        signals.append("Valid aperture")
    if "DateTimeOriginal" in exif: score += 0.08
    return round(score, 2), signals

def get_ai_suspicion_score(exif: dict, width: int = 0, height: int = 0, file_size: int = 0) -> tuple:
    score = 0.0
    signals = []
    has_camera_info = exif.get("Make") or exif.get("Model")
    ai_keywords = ["stable", "diffusion", "midjourney", "dalle", "flux", "sora", "generative"]
    software = str(exif.get("Software", "")).lower()
    make = str(exif.get("Make", "")).lower()
    if any(k in software for k in ai_keywords) or any(k in make for k in ai_keywords):
        score += 0.40
        signals.append("AI software signature")
    if not has_camera_info:
        score += 0.10
        signals.append("Missing hardware provenance")
    if width > 0 and height > 0 and not has_camera_info:
        if width in [512, 768, 1024, 1536, 2048] or height in [512, 768, 1024, 1536, 2048]:
            score += 0.15
            signals.append("AI-typical dimensions")
    return round(min(score, 1.0), 2), signals

async def detect_ai_media(file_path: str, trusted_metadata: dict = None, original_filename: str = None) -> dict:
    total_start = time.perf_counter()
    l1_data = {"status": "not_found", "provider": None, "description": "No cryptographic signature found."}
    t_c2pa = time.perf_counter()
    manifest = get_c2pa_manifest(file_path)
    if manifest:
        gen_info = manifest.get("claim_generator_info", [])
        generator = gen_info[0].get("name", "Unknown AI") if gen_info else manifest.get("claim_generator", "Unknown AI")
        is_ai = any("trainedAlgorithmicMedia" in str(a) for a in manifest.get("assertions", []))
        return {
            "summary": "Verified AI" if is_ai else "Verified Human",
            "confidence_score": 1.0,
            "suspicious": False,
            "layers": {
                "layer1_metadata": {
                    "status": "verified_ai" if is_ai else "verified_human", 
                    "provider": generator,
                    "description": "Media verified via C2PA manifest."
                },
                "layer2_forensics": {"status": "skipped"}
            },
            "gpu_bypassed": True
        }

    if file_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm')):
        # Video Logic
        video_metadata = await get_video_metadata(file_path)
        h_s, a_s, sigs, exit = get_video_metadata_score(video_metadata, os.path.basename(file_path), file_path)
        if exit: return {"summary": f"Verified {exit.capitalize()} Video", "confidence_score": 0.99}
        loop = asyncio.get_running_loop()
        frames, _ = await loop.run_in_executor(None, extract_video_frames, file_path)
        if not frames: return {"summary": "Analysis Failed", "confidence_score": 0.0}
        from app.runpod_client import run_batch_forensics
        batch_result = await run_batch_forensics(frames)
        probs = [r.get("ai_score", 0.0) for r in batch_result.get("results", [])]
        final_p = float(np.median(probs)) if probs else 0.0
        return {"summary": "Likely AI" if final_p > 0.5 else "Likely Human", "confidence_score": round(final_p, 2)}
    
    return await detect_ai_media_image_logic(file_path, l1_data, trusted_metadata=trusted_metadata, original_filename=original_filename)

async def detect_ai_media_image_logic(file_path: Optional[str], l1_data: dict = None, frame: Image.Image = None, trusted_metadata: dict = None, original_filename: str = None) -> dict:
    layer_start = time.perf_counter()
    if frame:
        img_pil = frame
        exif = {}
        width, height = img_pil.size
        file_size = 0
    else:
        exif = get_exif_data(file_path)
        img_pil = Image.open(file_path)
        width, height = img_pil.size
        file_size = os.path.getsize(file_path)

    if trusted_metadata:
        for k, v in trusted_metadata.items():
            if k in ["Make", "Model", "Software", "DateTime"]: exif[k] = v
        if "width" in trusted_metadata: width = trusted_metadata["width"]
        if "height" in trusted_metadata: height = trusted_metadata["height"]
        if "fileSize" in trusted_metadata: file_size = trusted_metadata["fileSize"]

    m_h, h_signals = get_forensic_metadata_score(exif)
    m_ai, ai_signals = get_ai_suspicion_score(exif, width, height, file_size)

    # --- 1. DUAL-GATE METADATA (RUN 30) ---
    if m_h > PROD_CONFIG['dual_gate_metadata']['human_threshold']:
        return {
            "summary": "Verified Human (Metadata)", "confidence_score": 1.0,
            "layers": {"layer1_metadata": {"status": "verified_human", "signals": h_signals}, "layer2_forensics": {"status": "skipped"}}
        }
    if m_ai > PROD_CONFIG['dual_gate_metadata']['ai_threshold']:
        return {
            "summary": "Verified AI (Metadata)", "confidence_score": 1.0,
            "layers": {"layer1_metadata": {"status": "verified_ai", "signals": ai_signals}, "layer2_forensics": {"status": "skipped"}}
        }

    # --- 2. GPU SCAN ---
    img_hash = get_image_hash(img_pil if frame else file_path, fast_mode=(frame is not None))
    cached = forensic_cache.get(img_hash)
    if cached:
        logger.info(f"[CACHE] Hit for {original_filename or file_path}")
        scores = cached["scores"]
        gpu_time = 0
    else:
        forensic_res = await run_deep_forensics(img_pil if frame else file_path, width, height)
        scores = forensic_res.get("scores", {"A": 0.5, "B": 0.5, "TruFor": 0.5})
        gpu_time = forensic_res.get("gpu_time_ms", 0)
        forensic_cache.put(img_hash, {"scores": scores})

    # --- 3. RUN 30 ENSEMBLE CONSENSUS ---
    ext = Path(file_path).suffix.lower() if file_path else ".jpg"
    pixels = width * height
    aspect = width / height if height != 0 else 1.0
    slice_name = get_slice_name(pixels, width, height, ext, aspect)
    p = PROD_CONFIG['slices'].get(slice_name, PROD_CONFIG['slices']['default'])

    l_a, l_b, l_t = logit(scores['A']), logit(scores['B']), logit(scores['TruFor'])
    l_total = (p['wA'] * l_a) + (p['wB'] * l_b) + (p['wT'] * l_t)
    
    l_final = l_total
    # Metadata Gating (Tau)
    if abs(l_total) < p['tau']:
        meta_signal = m_ai - m_h
        l_final = (p['alpha'] * l_total) + ((1 - p['alpha']) * meta_signal)

    # --- 4. FINAL VERDICT ---
    verdict_is_ai = l_final > p['margin']
    final_p = 1 / (1 + math.exp(-l_final))
    
    # Suspicion Window
    is_suspicious = abs(l_final - p['margin']) < PROD_CONFIG['suspicion']['window']
    
    summary = "Likely AI" if verdict_is_ai else "Likely Human"
    if is_suspicious: summary = "Suspicious (Uncertain)"
    
    return {
        "summary": summary,
        "confidence_score": round(final_p if verdict_is_ai else 1.0 - final_p, 2),
        "suspicious": is_suspicious,
        "layers": {
            "layer1_metadata": {"human_score": m_h, "ai_score": m_ai, "signals": ai_signals + h_signals},
            "layer2_forensics": {"status": "detected" if verdict_is_ai else "not_detected", "scores": scores, "slice": slice_name}
        },
        "gpu_time_ms": gpu_time
    }
