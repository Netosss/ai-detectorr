import runpod
import base64
import io
import time
import torch
import logging
import hashlib
import numpy as np
import cv2
import concurrent.futures
import os
import sys
from collections import OrderedDict
from PIL import Image, ImageFilter
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ---------------- Optimization Flags ----------------
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Device ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Initializing production worker on device: {device}")

# ---------------- Helper Functions ----------------
def apply_noise(img, sigma):
    img_np = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, img_np.shape).astype(np.float32)
    return Image.fromarray(np.clip(img_np + noise, 0, 255).astype(np.uint8))

def apply_sharpen(img, percent):
    return img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=percent, threshold=0))

def apply_upscale_sharpen(img, size=(224, 224), percent=110):
    img = img.resize(size, Image.Resampling.LANCZOS)
    return img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=percent, threshold=0))

# ---------------- Worker Cache ----------------
class WorkerLRUCache:
    def __init__(self, capacity: int = 500):
        self.cache = OrderedDict()
        self.capacity = capacity
    def get(self, key):
        if key not in self.cache: return None
        self.cache.move_to_end(key)
        return self.cache[key]
    def put(self, key, value):
        if key in self.cache: self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity: self.cache.popitem(last=False)

worker_cache = WorkerLRUCache(capacity=500)

# ---------------- TruFor Wrapper ----------------
class TruForWrapper:
    def __init__(self, device):
        self.device = device
        self.model = None
        self.load_model()

    def load_model(self):
        t_start = time.perf_counter()
        # Optimized path resolution for RunPod (Docker) and Local
        current_dir = Path(__file__).resolve().parent
        repo_root = Path("/app") if Path("/app").exists() else current_dir
        
        # --- Exhaustive search for TruFor lib ---
        possible_trufor_dirs = [
            repo_root / "third_party/grip/TruFor/TruFor_train_test",
            repo_root / "third_party/TruFor/TruFor_train_test",
            current_dir / "third_party/grip/TruFor/TruFor_train_test",
            Path("/app/third_party/grip/TruFor/TruFor_train_test"),
        ]
        
        trufor_path = None
        for d in possible_trufor_dirs:
            if (d / "lib").exists():
                trufor_path = d
                break
        
        # Fallback: Recursive search for the unique ph3 config file if initial search fails
        if not trufor_path:
            logger.info("Performing emergency recursive search for TruFor lib...")
            search_base = repo_root / "third_party"
            for p in search_base.rglob("trufor_ph3.yaml"):
                # The lib folder should be nearby: p is .../lib/config/trufor_ph3.yaml
                potential_root = p.parent.parent.parent
                if (potential_root / "lib").exists():
                    trufor_path = potential_root
                    logger.info(f"Emergency search found TruFor lib at: {trufor_path}")
                    break

        if not trufor_path:
            logger.error("Failed to locate TruFor lib in any expected path.")
            # Debug: List what we DO have
            try:
                if (repo_root / "third_party").exists():
                    logger.info(f"Contents of {repo_root / 'third_party'}:")
                    for item in (repo_root / "third_party").iterdir():
                        logger.info(f"  - {item}")
            except Exception as le:
                logger.error(f"Could not list directory: {le}")
            return

        if str(trufor_path) not in sys.path:
            sys.path.insert(0, str(trufor_path))
            logger.info(f"Initialized TruFor path: {trufor_path}")
            
        try:
            from lib.config import config as trufor_config
            from lib.utils import get_model
            config_file = trufor_path / "lib/config/trufor_ph3.yaml"
            
            # Find weights pathly
            model_file = None
            weights_roots = [trufor_path.parent / "pretrained_models", repo_root / "third_party/grip/TruFor/pretrained_models"]
            for wr in weights_roots:
                potential = wr / "weights/trufor.pth.tar"
                if potential.exists():
                    model_file = potential
                    break
            
            if not model_file:
                raise FileNotFoundError("Could not find trufor.pth.tar weights file.")
            
            cfg = trufor_config
            cfg.defrost()
            cfg.merge_from_file(str(config_file))
            
            # Standardize internal weight paths (assuming they are in TruFor_train_test/pretrained_models)
            base_pretrained = trufor_path / "pretrained_models"
            cfg.MODEL.EXTRA.NOISEPRINT = str(base_pretrained / "noiseprint++/noiseprint++.th")
            cfg.MODEL.EXTRA.SEGFORMER = str(base_pretrained / "segformers/mit_b2.pth")
            cfg.freeze()
            
            self.model = get_model(cfg)
            checkpoint = torch.load(model_file, map_location=torch.device(self.device), weights_only=False)
            self.model.load_state_dict(checkpoint.get('state_dict', checkpoint))
            self.model = self.model.to(self.device).eval()
            
            duration = (time.perf_counter() - t_start) * 1000
            logger.info(f"TruFor Loaded Successfully in {duration:.2f}ms")
        except Exception as e:
            logger.error(f"Failed to load TruFor: {e}", exc_info=True)

    @torch.no_grad()
    def predict(self, img_pil_list):
        if self.model is None: return [0.5] * len(img_pil_list)
        
        # Optimize for 4090: Batch TruFor inference if multiple images
        try:
            batch_tensors = []
            for img_pil in img_pil_list:
                # Resize to 1024 max for TruFor stability on GPU
                if max(img_pil.size) > 1024:
                    img_pil = img_pil.copy()
                    img_pil.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                
                img_np = np.array(img_pil).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
                batch_tensors.append(img_tensor)

            # Note: TruFor usually expects same-size images in a batch. 
            # If they differ, we process sequentially or pad. 
            # Since these are surgically preprocessed and might differ in size, 
            # we keep sequential but optimize the tensor transfer.
            results = []
            for t in batch_tensors:
                t = t.unsqueeze(0).to(self.device, non_blocking=True)
                _, _, det, _ = self.model(t)
                results.append(torch.sigmoid(det).item() if det is not None else 0.5)
            return results
        except Exception as e:
            logger.error(f"TruFor inference error: {e}")
            return [0.5] * len(img_pil_list)

# ---------------- Router Classifier ----------------
class RouterClassifier:
    def __init__(self):
        self.device = device
        self.models_loaded = False
        # Thread pool for inference models (3 models now)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.load_models()

    def load_models(self):
        if self.models_loaded: return
        logger.info("Loading Production Models (Run 30 Config)...")
        try:
            # Model A
            self.processor_a = AutoImageProcessor.from_pretrained("haywoodsloan/ai-image-detector-dev-deploy", use_fast=True)
            # Use float16 ONLY on CUDA for speed, float32 on CPU/MPS for stability
            model_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.model_a = AutoModelForImageClassification.from_pretrained(
                "haywoodsloan/ai-image-detector-dev-deploy", torch_dtype=model_dtype
            ).to(self.device).eval()

            # Model B
            self.processor_b = AutoImageProcessor.from_pretrained("Ateeqq/ai-vs-human-image-detector", use_fast=True)
            self.model_b = AutoModelForImageClassification.from_pretrained(
                "Ateeqq/ai-vs-human-image-detector", torch_dtype=model_dtype
            ).to(self.device).eval()

            # TruFor
            self.trufor = TruForWrapper(self.device)

            # Warmup (IMPORTANT: Must happen after any compilation or optimization)
            logger.info("Running model warmup...")
            dummy = Image.new('RGB', (224, 224), color='white')
            # Run sequentially for warmup to ensure each model is initialized
            self._predict_single(self.model_a, self.processor_a, [dummy])
            self._predict_single(self.model_b, self.processor_b, [dummy])
            self.trufor.predict([dummy])

            # NOTE: torch.compile is removed for stability. 
            # First-run compilation overhead was exceeding RunPod timeouts.

            self.models_loaded = True
            logger.info("All Production Models Loaded Successfully.")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def _get_ai_idx(self, model):
        for idx, label in model.config.id2label.items():
            if any(k in label.lower() for k in ['ai', 'fake', 'generated']):
                return idx
        return 0

    @torch.no_grad()
    def _predict_single(self, model, processor, images):
        try:
            # Use fast path for tensor conversion
            inputs = processor(images=images, return_tensors="pt")
            
            # Transfer to GPU with non_blocking=True
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    # channels_last is only for 4D tensors (B, C, H, W)
                    if v.ndim == 4:
                        inputs[k] = v.to(self.device, non_blocking=True, memory_format=torch.channels_last)
                    else:
                        inputs[k] = v.to(self.device, non_blocking=True)
                        
                    if inputs[k].dtype == torch.float32 and getattr(model, "dtype", torch.float32) == torch.float16:
                        inputs[k] = inputs[k].half()
            
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            ai_idx = self._get_ai_idx(model)
            return probs[:, ai_idx].cpu().numpy()
        except Exception as e:
            logger.error(f"Inference error in _predict_single: {e}", exc_info=True)
            return [0.5] * len(images)

    def predict_batch(self, images: list, is_png_list: list = None):
        """
        Implements Run 30 Surgical Preprocessing and Model Inference.
        Returns raw scores for A, B, and TruFor.
        """
        batch_size = len(images)
        batch_a = []
        batch_b = []
        batch_t = []
        
        for idx, img in enumerate(images):
            img = img.convert("RGB")
            w, h = img.size
            pixels = w * h
            is_png = is_png_list[idx] if is_png_list else False
            
            # --- Model A Surgical ---
            if is_png: img_a = apply_noise(img, 5)
            elif pixels < 2000: img_a = apply_noise(img, 10)
            elif 2000 <= pixels < 10000: img_a = apply_noise(img, 5)
            elif 10000 <= pixels < 500000:
                t_size = (224, 224) if pixels > 100000 else (448, 448)
                img_a = apply_upscale_sharpen(img, size=t_size, percent=110)
            else: img_a = apply_sharpen(img, 110)
            batch_a.append(img_a)
            
            # --- Model B Surgical ---
            if pixels < 2000: img_b = apply_noise(img, 5)
            elif 2000 <= pixels < 10000: img_b = apply_noise(img, 3)
            else: img_b = img # RAW for >10k
            batch_b.append(img_b)
            
            # --- TruFor Surgical ---
            img_t = img
            if pixels >= 200000: img_t = apply_sharpen(img, 110)
            batch_t.append(img_t)

        # Launch all 3 models in parallel with timing
        logger.info(f"Starting inference batch (size={batch_size})...")
        t_inf_start = time.perf_counter()
        
        f_a = self.executor.submit(self._predict_single, self.model_a, self.processor_a, batch_a)
        f_b = self.executor.submit(self._predict_single, self.model_b, self.processor_b, batch_b)
        f_t = self.executor.submit(self.trufor.predict, batch_t)
        
        # We use a timeout to prevent absolute "stuck" states
        # Increased to 30s to account for potentially hardware initialization on cold start
        done, not_done = concurrent.futures.wait([f_a, f_b, f_t], timeout=30.0)
        
        if not_done:
            logger.warning(f"Inference timed out for {len(not_done)} models!")
            for f in not_done:
                # We can't easily kill threads, but we can log which one failed
                if f == f_a: logger.error("Model A hang detected.")
                elif f == f_b: logger.error("Model B hang detected.")
                elif f == f_t: logger.error("TruFor hang detected.")

        res_a = f_a.result() if f_a.done() else [0.5] * batch_size
        res_b = f_b.result() if f_b.done() else [0.5] * batch_size
        res_t = f_t.result() if f_t.done() else [0.5] * batch_size
        
        inf_duration = (time.perf_counter() - t_inf_start) * 1000
        logger.info(f"Inference batch completed in {inf_duration:.2f}ms")
        
        final_results = []
        for i in range(batch_size):
            final_results.append({
                "A": float(res_a[i]),
                "B": float(res_b[i]),
                "TruFor": float(res_t[i])
            })
        return final_results

from pathlib import Path
classifier = RouterClassifier()
decode_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

def decode_image(args):
    idx, b64_str = args
    try:
        img_bytes = base64.b64decode(b64_str)
        img_hash = hashlib.md5(img_bytes).hexdigest()
        return (idx, img_bytes, img_hash, None)
    except Exception as e:
        return (idx, None, None, str(e))

def handler(job):
    job_input = job.get("input", {})
    images_b64 = []
    is_batch = False
    
    if "images" in job_input and isinstance(job_input["images"], list):
        images_b64 = job_input["images"]
        is_batch = True
    elif "image" in job_input:
        images_b64 = [job_input["image"]]
    else:
        return {"error": "No image data provided"}

    total_start = time.perf_counter()
    results = [None] * len(images_b64)
    
    images_to_process = []
    hashes_to_process = []
    
    t0 = time.perf_counter()
    decode_args = [(i, s) for i, s in enumerate(images_b64)]
    futures = decode_pool.map(decode_image, decode_args)
    
    for idx, img_bytes, img_hash, err in futures:
        if err:
            results[idx] = {"error": err, "scores": {"A": 0.5, "B": 0.5, "TruFor": 0.5}}
            continue
            
        cached = worker_cache.get(img_hash)
        if cached:
            res = cached.copy()
            res["cache_hit"] = True
            results[idx] = res
            continue
            
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            images_to_process.append((idx, img))
            hashes_to_process.append(img_hash)
        except Exception as e:
             results[idx] = {"error": str(e), "scores": {"A": 0.5, "B": 0.5, "TruFor": 0.5}}

    decode_ms = (time.perf_counter() - t0) * 1000
    
    if images_to_process:
        pil_images = [x[1] for x in images_to_process]
        is_png_list = [job_input.get("is_png", False)] * len(pil_images)
        t1 = time.perf_counter()
        
        raw_scores = classifier.predict_batch(pil_images, is_png_list=is_png_list)
        
        gpu_ms = (time.perf_counter() - t1) * 1000
        
        for i, scores in enumerate(raw_scores):
            original_idx = images_to_process[i][0]
            res = {"scores": scores, "cache_hit": False}
            # The app layer expects 'ai_score' for backwards compatibility if needed, 
            # but we'll primarily use the breakdown.
            # Calculating a simple average as fallback ai_score
            res["ai_score"] = (scores["A"] + scores["B"] + scores["TruFor"]) / 3
            worker_cache.put(hashes_to_process[i], res)
            results[original_idx] = res
                
    total_ms = (time.perf_counter() - total_start) * 1000
    
    response = {
        "results": results if is_batch else results[0],
        "timing_ms": {
            "decode": round(decode_ms, 2),
            "total": round(total_ms, 2)
        }
    }
    
    if not is_batch:
        return {**results[0], "timing_ms": response["timing_ms"]}
        
    return response

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
