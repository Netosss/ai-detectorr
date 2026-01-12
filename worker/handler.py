import runpod
import base64
import io
import time
import torch
import logging
import hashlib
import numpy as np
import concurrent.futures
import os
<<<<<<< HEAD
import sys
=======
import json
>>>>>>> 55b8f71 (empty commit)
from pathlib import Path
from collections import OrderedDict
from PIL import Image, ImageFilter
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL.ExifTags import TAGS

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
    return np.log(p / (1.0 - p))

def get_slice_name(pixels):
    if pixels < 2000: return "thumbnail"
    if 2000 <= pixels < 10000: return "low_res"
    if 10000 <= pixels < 50000: return "10k-50k"
    if 50000 <= pixels < 500000: return "50k-500k"
    return ">500k"

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

<<<<<<< HEAD
    def load_model(self):
        t_start = time.perf_counter()
        
        # --- Direct Path Resolution (Optimized for RunPod) ---
        repo_root = Path("/app")
        
        # Try multiple possible path structures (Docker COPY can be tricky)
        possible_paths = [
            repo_root / "third_party/grip/TruFor/TruFor_train_test",
            repo_root / "grip/TruFor/TruFor_train_test",
            Path(__file__).resolve().parent / "third_party/grip/TruFor/TruFor_train_test"
        ]
        
        trufor_path = None
        for p in possible_paths:
            if (p / "lib").exists():
                trufor_path = p
                break
            
        if not trufor_path:
            logger.error(f"❌ TruFor path not found. Checked: {[str(p) for p in possible_paths]}")
            # Debug: what is in /app?
            try:
                if repo_root.exists():
                    logger.info(f"Contents of /app: {[str(x.name) for x in repo_root.iterdir()]}")
            except: pass
            return

        # Add both the folder and its parent to be safe
        for p in [str(trufor_path), str(trufor_path.parent)]:
            if p not in sys.path:
                sys.path.insert(0, p)
            
        try:
            # We use absolute imports by ensuring the path is correct
            import lib.config
            import lib.utils
            
            trufor_config = lib.config.config
            get_model = lib.utils.get_model
            
            config_file = trufor_path / "lib/config/trufor_ph3.yaml"
            model_file = repo_root / "third_party/grip/TruFor/pretrained_models/weights/trufor.pth.tar"
            
            if not model_file.exists():
                model_file = trufor_path.parent / "pretrained_models/weights/trufor.pth.tar"
            
            cfg = trufor_config
            cfg.defrost()
            cfg.merge_from_file(str(config_file))
            
            # Direct weight paths
            base_pretrained = trufor_path.parent / "pretrained_models"
            cfg.MODEL.EXTRA.NOISEPRINT = str(base_pretrained / "noiseprint++/noiseprint++.th")
            cfg.MODEL.EXTRA.SEGFORMER = str(base_pretrained / "segformers/mit_b2.pth")
            cfg.freeze()
            
            self.model = get_model(cfg)
            # Use mmap=True for instant loading from disk
            checkpoint = torch.load(model_file, map_location=torch.device(self.device), weights_only=False, mmap=True)
            self.model.load_state_dict(checkpoint.get('state_dict', checkpoint))
            self.model = self.model.to(self.device).eval()
            
            duration = (time.perf_counter() - t_start) * 1000
            logger.info(f"✅ TruFor Ready (Direct Load) in {duration:.2f}ms")
        except Exception as e:
            logger.error(f"❌ Failed to load TruFor: {e}")
            import traceback
            logger.error(traceback.format_exc())

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
        self.skip_trufor = os.environ.get("SKIP_TRUFOR", "false").lower() == "true"
        # Thread pool for inference models (3 models now)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.load_models()

    def load_models(self):
        if self.models_loaded: return
        t_boot_start = time.perf_counter()
        
        mode_str = "LITE (No TruFor)" if self.skip_trufor else "FULL"
        logger.info(f"🚀 Booting Production Worker in {mode_str} mode...")
        
        try:
            # --- Parallel Loading ---
            # We load Model A, Model B, and TruFor simultaneously to slash cold-start time
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as loader:
                f_a = loader.submit(self._init_model_a)
                f_b = loader.submit(self._init_model_b)
                
                if not self.skip_trufor:
                    f_t = loader.submit(lambda: TruForWrapper(self.device))
                else:
                    f_t = None
                
                wait_list = [f_a, f_b]
                if f_t: wait_list.append(f_t)
                
                concurrent.futures.wait(wait_list)
                
                self.model_a, self.processor_a = f_a.result()
                self.model_b, self.processor_b = f_b.result()
                self.trufor = f_t.result() if f_t else None

            # --- Optimizations ---
            # NOTE: torch.compile is disabled for Model A/B due to FX conflict with transformers 4.48+
            # on PyTorch 2.2.1. FP16 + channels_last provides enough speed on RTX 4090.
            if hasattr(torch, 'compile') and self.device == "cuda":
                logger.info("🛠 Skipping torch.compile to avoid FX conflict (stability fix).")
                # try:
                #     self.model_a = torch.compile(self.model_a, mode="default")
                #     self.model_b = torch.compile(self.model_b, mode="default")
                # except Exception as ce:
                #     logger.warning(f"Could not compile: {ce}")

            # Warmup (IMPORTANT: Moves 4090 initialization to boot phase)
            logger.info("🔥 RTX 4090: Hard-warming inference engines...")
            dummy = Image.new('RGB', (224, 224), color='white')
            # 1 pass is enough to initialize CUDA context and cuDNN
            self.predict_batch([dummy])
            
            boot_ms = (time.perf_counter() - t_boot_start) * 1000
            self.models_loaded = True
            logger.info(f"✨ Production Worker Ready in {boot_ms:.2f}ms")
            
        except Exception as e:
            logger.error(f"FATAL: Failed to boot worker: {e}", exc_info=True)
            raise

    def _init_model_a(self):
        mid = "haywoodsloan/ai-image-detector-dev-deploy"
        proc = AutoImageProcessor.from_pretrained(mid, use_fast=False)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model = AutoModelForImageClassification.from_pretrained(mid, torch_dtype=dtype).to(self.device).eval()
        if self.device == "cuda":
            model = model.half().to(memory_format=torch.channels_last)
        return model, proc

    def _init_model_b(self):
        mid = "Ateeqq/ai-vs-human-image-detector"
        proc = AutoImageProcessor.from_pretrained(mid, use_fast=False)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model = AutoModelForImageClassification.from_pretrained(mid, torch_dtype=dtype).to(self.device).eval()
        if self.device == "cuda":
            model = model.half().to(memory_format=torch.channels_last)
        return model, proc

    def _get_ai_idx(self, model):
        for idx, label in model.config.id2label.items():
            if any(k in label.lower() for k in ['ai', 'fake', 'generated']):
                return idx
        return 0
=======
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
>>>>>>> 55b8f71 (empty commit)

    @torch.no_grad()
    def _predict_model(self, mid, images):
        model, processor, prep = self.models[mid], self.processors[mid], self.best_preprocess[mid]
        processed_imgs = []
        for img in images:
            if prep["upscale"]: img = apply_upscale(img, prep["upscale"])
            if prep["sharpen"] > 100: img = apply_sharpen(img, prep["sharpen"])
            processed_imgs.append(img)
            
<<<<<<< HEAD
            # Get the model's actual parameter dtype to avoid Half/Float mismatch
            model_dtype = next(model.parameters()).dtype
            
            # Transfer to GPU with non_blocking=True
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(self.device, non_blocking=True)
                    # channels_last is only for 4D tensors (B, C, H, W)
                    if v.ndim == 4:
                        v = v.to(memory_format=torch.channels_last)
                    
                    if v.is_floating_point():
                        v = v.to(model_dtype)
                    
                    inputs[k] = v
            
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            ai_idx = self._get_ai_idx(model)
            return probs[:, ai_idx].cpu().numpy()
        except Exception as e:
            logger.error(f"Inference error in _predict_single: {e}", exc_info=True)
            return [0.5] * len(images)
=======
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
        return probs[:, self.ai_indices[mid]].cpu().numpy()
>>>>>>> 55b8f71 (empty commit)

    def predict_batch(self, images_data: list):
        # images_data contains (PIL_Image, EXIF_Dict)
        batch_size = len(images_data)
        pil_images = [x[0] for x in images_data]
        
        # CPU parallel execution for model predictions
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(self._predict_model, mid, pil_images): mid for mid in self.models.keys()}
            results_by_model = {futures[f]: f.result() for f in concurrent.futures.as_completed(futures)}
            
        final_batch_results = []
        window = self.config['suspicion']['window']
        
<<<<<<< HEAD
        f_a = self.executor.submit(self._predict_single, self.model_a, self.processor_a, batch_a)
        f_b = self.executor.submit(self._predict_single, self.model_b, self.processor_b, batch_b)
        
        if self.trufor:
            f_t = self.executor.submit(self.trufor.predict, batch_t)
        else:
            f_t = None
        
        wait_list = [f_a, f_b]
        if f_t: wait_list.append(f_t)
        
        # We use a timeout to prevent absolute "stuck" states
        done, not_done = concurrent.futures.wait(wait_list, timeout=30.0)
        
        if not_done:
            logger.warning(f"Inference timed out for {len(not_done)} models!")
            for f in not_done:
                if f == f_a: logger.error("Model A hang detected.")
                elif f == f_b: logger.error("Model B hang detected.")
                elif f == f_t: logger.error("TruFor hang detected.")

        res_a = f_a.result() if f_a.done() else [0.5] * batch_size
        res_b = f_b.result() if f_b.done() else [0.5] * batch_size
        res_t = f_t.result() if (f_t and f_t.done()) else [0.5] * batch_size
        
        inf_duration = (time.perf_counter() - t_inf_start) * 1000
        logger.info(f"Inference batch completed in {inf_duration:.2f}ms")
        
        final_results = []
=======
>>>>>>> 55b8f71 (empty commit)
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
            
            margin = slice_cfg['margin']
            is_ai = l_total > margin
            
            # Magic Tool Logic
            magic_triggered = False
            if (px > 500000 and abs(l_total - margin) < window) or \
               (max(model_probs.values()) - min(model_probs.values()) > 0.8) or \
               (m_h > 0.7 and l_total > (margin + 1.5)):
                magic_triggered, is_ai = True, True
            
            final_batch_results.append({
                "ai_score": float(torch.sigmoid(torch.tensor(l_total)).item()),
                "is_ai": bool(is_ai),
                "magic_triggered": magic_triggered,
                "slice": slice_name,
                "breakdown": model_probs,
                "metadata_h": m_h
            })
        return final_batch_results

<<<<<<< HEAD
classifier = RouterClassifier()
=======
classifier = EnsembleClassifier()
>>>>>>> 55b8f71 (empty commit)
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
