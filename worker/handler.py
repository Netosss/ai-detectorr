import runpod
import base64
import io
import time
import torch
import logging
import hashlib
import concurrent.futures
import os
import json
from PIL import Image, ImageFilter
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL.ExifTags import TAGS
import math

# ---------------- Optimization Flags (RTX 4090 Optimized) ----------------
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Pre-processing Helpers ----------------
def apply_sharpen(img, percent):
    return img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=percent, threshold=0))

def apply_upscale(img, size):
    w, h = img.size
    if w >= h: new_w, new_h = size, int(h * (size / w))
    else: new_h, new_w = size, int(w * (size / h))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

def logit(p):
    p = max(min(p, 0.999999), 0.000001)
    return math.log(p / (1.0 - p))

def get_slice_name(pixels):
    if pixels < 2000: return "thumbnail"
    if 2000 <= pixels < 10000: return "low_res"
    if 10000 <= pixels < 50000: return "10k-50k"
    if 50000 <= pixels < 500000: return "50k-500k"
    return ">500k"

def should_use_magic_tool(px, l_total, margin, window, model_probs, m_h):
    """
    Decides whether to trigger the 'Magic Tool' override.
    Currently separated as per user request to be toggled manually.
    """
    # Rule 1: High-Res Uncertainty (near the margin)
    if px > 500000 and abs(l_total - margin) < window:
        return True
    
    # Rule 2: Inter-model Conflict (massive disagreement between models)
    if (max(model_probs.values()) - min(model_probs.values())) > 0.8:
        return True
        
    # Rule 3: Metadata Paradox (high human signals but high ensemble AI logit)
    if m_h > 0.7 and l_total > (margin + 1.5):
        return True
        
    return False

# ---------------- Metadata Scoring ----------------
def get_forensic_metadata_score(exif: dict) -> float:
    score = 0.0
    def to_float(val):
        try: return float(val)
        except: return None
    make = str(exif.get("Make", "")).lower()
    software = str(exif.get("Software", "")).lower()
    if any(m in make for m in ["apple", "google", "samsung", "sony", "canon", "nikon"]): score += 0.35
    if any(s in software for s in ["ios", "android", "lightroom"]): score += 0.25
    exp = to_float(exif.get("ExposureTime"))
    if exp is not None and 0 < exp < 30: score += 0.15
    iso = to_float(exif.get("ISOSpeedRatings"))
    if iso is not None and 50 <= iso <= 102400: score += 0.15
    f_num = to_float(exif.get("FNumber"))
    if f_num is not None and 0.95 <= f_num <= 32: score += 0.15
    if "DateTimeOriginal" in exif: score += 0.08
    return round(score, 2)

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

# ---------------- Ensemble Classifier ----------------
class EnsembleClassifier:
    def __init__(self):
        self.device = device
        self.config = self._load_config()
        self.models = {}
        self.processors = {}
        self.ai_indices = {
            "haywoodsloan/ai-image-detector-dev-deploy": 0,
            "Ateeqq/ai-vs-human-image-detector": 0,
            "Organika/sdxl-detector": 0
        }
        self.best_preprocess = {
            "haywoodsloan/ai-image-detector-dev-deploy": {"sharpen": 110, "upscale": None},
            "Ateeqq/ai-vs-human-image-detector": {"sharpen": 110, "upscale": None},
            "Organika/sdxl-detector": {"sharpen": 100, "upscale": 224}
        }
        self.load_models_parallel()

    def _load_config(self):
        paths = ["/app/configs/production_v1.json", "configs/new_optimized_config.json"]
        for p in paths:
            if os.path.exists(p):
                with open(p, 'r') as f: return json.load(f)
        raise FileNotFoundError("Production config missing!")

    def _init_single_model(self, mid):
        logger.info(f"  Loading {mid}...")
        proc = AutoImageProcessor.from_pretrained(mid, use_fast=False)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model = AutoModelForImageClassification.from_pretrained(mid, torch_dtype=dtype).to(self.device).eval()
        if self.device == "cuda":
            model = model.half().to(memory_format=torch.channels_last)
        return mid, model, proc

    def load_models_parallel(self):
        t_start = time.perf_counter()
        logger.info("🚀 Parallel Boot: Loading Winning Ensemble...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(self._init_single_model, mid) for mid in self.best_preprocess.keys()]
            for future in concurrent.futures.as_completed(futures):
                mid, model, proc = future.result()
                self.models[mid] = model
                self.processors[mid] = proc

        logger.info(f"✨ Production Ready in {(time.perf_counter() - t_start)*1000:.2f}ms")

    @torch.no_grad()
    def _predict_model(self, mid, images):
        model, processor, prep = self.models[mid], self.processors[mid], self.best_preprocess[mid]
        processed_imgs = []
        for img in images:
            if prep["upscale"]: img = apply_upscale(img, prep["upscale"])
            if prep["sharpen"] > 100: img = apply_sharpen(img, prep["sharpen"])
            processed_imgs.append(img)
            
        inputs = processor(images=processed_imgs, return_tensors="pt").to(self.device)
        model_dtype = next(model.parameters()).dtype
        
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.to(self.device, non_blocking=True)
                if v.ndim == 4: v = v.to(memory_format=torch.channels_last)
                if v.is_floating_point(): v = v.to(model_dtype)
                inputs[k] = v
                
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs[:, self.ai_indices[mid]].cpu().tolist()

    def predict_batch(self, images_data: list):
        # images_data contains (PIL_Image, EXIF_Dict)
        batch_size = len(images_data)
        pil_images = [x[0] for x in images_data]
        
        # parallel execution for model predictions
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(self._predict_model, mid, pil_images): mid for mid in self.models.keys()}
            results_by_model = {futures[f]: f.result() for f in concurrent.futures.as_completed(futures)}
            
        final_batch_results = []
        window = self.config['suspicion']['window']
        
        for i in range(batch_size):
            img, exif = images_data[i]
            px = img.size[0] * img.size[1]
            slice_name = get_slice_name(px)
            slice_cfg = self.config['slices'].get(slice_name, self.config['slices']['>500k'])
            
            m_h = get_forensic_metadata_score(exif)
            
            l_total = 0
            model_probs = {}
            for mid in self.models.keys():
                p = float(results_by_model[mid][i])
                model_probs[mid.split('/')[-1]] = p
                l_total += slice_cfg['weights'].get(mid, 0.0) * logit(p)
            
            # --- Metadata Gating (Run 30 logic) ---
            m_ai, _ = get_ai_suspicion_score(exif, img.size[0], img.size[1], 0) # file_size unknown
            l_final = l_total
            if abs(l_total) < slice_cfg.get('tau', 0.0):
                alpha = slice_cfg.get('alpha', 1.0)
                meta_signal = m_ai - m_h
                l_final = (alpha * l_total) + ((1 - alpha) * meta_signal)

            margin = slice_cfg['margin']
            is_ai = l_final > margin
            
            # Suspicion Window logic
            is_suspicious = abs(l_final - margin) < window
            
            # Magic Tool Logic (Extracted but disabled for now)
            # magic_triggered = should_use_magic_tool(px, l_final, margin, window, model_probs, m_h)
            # if magic_triggered: is_ai = True
            magic_triggered = False 
            
            final_batch_results.append({
                "ai_score": float(torch.sigmoid(torch.tensor(l_final)).item()),
                "is_ai": bool(is_ai),
                "suspicious": bool(is_suspicious),
                "magic_triggered": magic_triggered,
                "slice": slice_name,
                "breakdown": model_probs,
                "metadata_h": m_h,
                "metadata_ai": m_ai
            })
        return final_batch_results

classifier = EnsembleClassifier()
decode_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

def decode_and_meta(args):
    idx, b64_str = args
    try:
        img_bytes = base64.b64decode(b64_str)
        img_hash = hashlib.md5(img_bytes).hexdigest()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        # Extract EXIF during decode phase (Offloads from main prediction loop)
        exif = img._getexif() or {}
        exif_data = {TAGS.get(tag, tag): val for tag, val in exif.items()}
        return (idx, img, exif_data, img_hash, None)
    except Exception as e:
        return (idx, None, None, None, str(e))

def handler(job):
    job_input = job.get("input", {})
    images_b64 = job_input.get("images", [job_input.get("image")]) if "image" in job_input or "images" in job_input else []
    if not images_b64 or images_b64[0] is None: return {"error": "No image data"}

    total_start = time.perf_counter()
    results = [None] * len(images_b64)
    images_to_process = [] # Will hold (idx, PIL_Image, Exif_Dict, Hash)
    
    # 1. Parallel Decode & Metadata Extraction
    decode_args = [(i, s) for i, s in enumerate(images_b64)]
    for idx, img, exif, img_hash, err in decode_pool.map(decode_and_meta, decode_args):
        if err:
            results[idx] = {"error": err}
            continue
        images_to_process.append((idx, img, exif, img_hash))

    # 2. Batch GPU Inference
    if images_to_process:
        batch_input = [(x[1], x[2]) for x in images_to_process]
        batch_results = classifier.predict_batch(batch_input)
        for i, res in enumerate(batch_results):
            results[images_to_process[i][0]] = res

    return {
        "results": results if isinstance(job_input.get("images"), list) else results[0],
        "timing_ms": round((time.perf_counter() - total_start) * 1000, 2)
    }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
