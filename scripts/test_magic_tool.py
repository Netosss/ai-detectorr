import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from PIL.ExifTags import TAGS
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from typing import Dict, List, Any

# --- Configuration ---
CONFIG_PATH = "configs/new_optimized_config.json"
CACHE_PATH = "reports/test_evaluation_data.json"
TEST_AI_DIR = "/Users/netanel.ossi/Desktop/ai_detector_test_set/ai"
TEST_REAL_DIR = "/Users/netanel.ossi/Desktop/ai_detector_test_set/real"

BEST_PREPROCESS = {
    "Ateeqq/ai-vs-human-image-detector": {"sharpen": 110, "upscale": None},
    "haywoodsloan/ai-image-detector-dev-deploy": {"sharpen": 110, "upscale": None},
    "Organika/sdxl-detector": {"sharpen": 100, "upscale": 224}
}

# --- Metadata Helpers ---
def get_exif_data(file_path: str) -> dict:
    try:
        with Image.open(file_path) as img:
            exif = img._getexif() or {}
            exif_data = {}
            for tag, value in exif.items():
                decoded = TAGS.get(tag, tag)
                exif_data[decoded] = value
            return exif_data
    except:
        return {}

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

def logit(p):
    p = max(min(p, 0.999999), 0.000001)
    return np.log(p / (1.0 - p))

def get_slice_name(pixels):
    if pixels < 2000: return "thumbnail"
    if 2000 <= pixels < 10000: return "low_res"
    if 10000 <= pixels < 50000: return "10k-50k"
    if 50000 <= pixels < 500000: return "50k-500k"
    return ">500k"

def apply_sharpen(img: Image.Image, percent: int) -> Image.Image:
    return img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=percent, threshold=0))

def apply_upscale(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    if w >= h:
        new_w = size
        new_h = int(h * (size / w))
    else:
        new_h = size
        new_w = int(w * (size / h))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

class AIModel:
    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device
        self.dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_id, torch_dtype=self.dtype
        ).to(self.device).eval()
        
        self.ai_idx = 0
        for idx, label in self.model.config.id2label.items():
            if any(k in label.lower() for k in ['ai', 'artificial', 'fake', 'generated']):
                self.ai_idx = idx
                break

    @torch.no_grad()
    def predict(self, images: List[Image.Image]) -> np.ndarray:
        if not images: return np.array([])
        batch_size = 32
        all_probs = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            if self.device in ["cuda", "mps"]:
                inputs = {k: v.to(self.dtype) if v.is_floating_point() else v for k, v in inputs.items()}
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            all_probs.append(probs[:, self.ai_idx].cpu().numpy())
        return np.concatenate(all_probs)

def should_use_magic_tool(pixels, l_total, margin, window, model_probs, m_h):
    """Magic Tool Trigger Logic"""
    # Rule 1: High-Res Uncertainty
    if pixels > 500000 and abs(l_total - margin) < window:
        return True
    
    # Rule 2: Big Conflict (>0.8 difference between models)
    if (max(model_probs) - min(model_probs)) > 0.8:
        return True
        
    # Rule 3: Metadata Paradox
    if m_h > 0.7 and l_total > (margin + 1.5):
        return True
        
    return False

def main():
    # 1. Load Config
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    # 2. Check for Cache
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached evaluation data from {CACHE_PATH}...")
        with open(CACHE_PATH, 'r') as f:
            eval_data = json.load(f)
        df = pd.DataFrame(eval_data['images'])
        model_ids = eval_data['model_ids']
    else:
        # Run full inference once
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Cache not found. Running full inference on device: {device}")

        data = []
        for folder, label in [(TEST_AI_DIR, 1), (TEST_REAL_DIR, 0)]:
            if os.path.exists(folder):
                files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for f in files:
                    data.append({'path': os.path.join(folder, f), 'label_true': label})
        
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} images from test set.")

        model_ids = list(BEST_PREPROCESS.keys())
        models = {mid: AIModel(mid, device) for mid in model_ids}

        image_data_list = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
            path = row['path']
            try:
                img = Image.open(path).convert("RGB")
                pixels = img.size[0] * img.size[1]
                
                exif = get_exif_data(path)
                m_h = get_forensic_metadata_score(exif)
                
                model_probs = {}
                for mid, model in models.items():
                    prep = BEST_PREPROCESS[mid]
                    processed_img = img
                    if prep['upscale']: processed_img = apply_upscale(processed_img, prep['upscale'])
                    if prep['sharpen'] > 100: processed_img = apply_sharpen(processed_img, prep['sharpen'])
                    
                    prob = float(model.predict([processed_img])[0])
                    model_probs[mid] = prob
                
                image_data_list.append({
                    "path": path,
                    "label_true": row['label_true'],
                    "pixels": pixels,
                    "m_h": m_h,
                    "model_probs": model_probs
                })
            except Exception as e:
                print(f"Error processing {path}: {e}")

        # Save Cache
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        eval_data = {"model_ids": model_ids, "images": image_data_list}
        with open(CACHE_PATH, 'w') as f:
            json.dump(eval_data, f)
        print(f"Saved evaluation data to {CACHE_PATH}")
        df = pd.DataFrame(image_data_list)

    # 3. Apply Comparison Logic
    print("\nComparing Standard Ensemble vs Magic Tool Override...")
    
    df['slice'] = df['pixels'].apply(get_slice_name)
    window = config['suspicion']['window']
    
    results = []
    magic_triggers = 0
    
    for idx, row in df.iterrows():
        slice_name = row['slice']
        slice_cfg = config['slices'].get(slice_name, config['slices']['>500k'])
        
        # 1. Standard Ensemble
        probs = [row['model_probs'][mid] for mid in model_ids]
        logits = [logit(p) for p in probs]
        
        l_total = 0
        for i, mid in enumerate(model_ids):
            weight = slice_cfg['weights'].get(mid, 0.0)
            l_total += weight * logits[i]
            
        margin = slice_cfg['margin']
        std_pred = 1 if l_total > margin else 0
        
        # 2. Magic Tool Logic
        is_triggered = should_use_magic_tool(row['pixels'], l_total, margin, window, probs, row['m_h'])
        magic_pred = 1 if is_triggered else std_pred
        if is_triggered: magic_triggers += 1
        
        results.append({
            "path": row['path'],
            "label_true": row['label_true'],
            "pred_std": std_pred,
            "pred_magic": magic_pred,
            "triggered": is_triggered,
            "l_total": l_total,
            "margin": margin
        })

    results_df = pd.DataFrame(results)
    y_true = results_df['label_true'].values
    
    # Metrics
    ba_std = balanced_accuracy_score(y_true, results_df['pred_std'])
    tn_s, fp_std, fn_std, tp_s = confusion_matrix(y_true, results_df['pred_std'], labels=[0, 1]).ravel()
    
    ba_magic = balanced_accuracy_score(y_true, results_df['pred_magic'])
    tn_m, fp_magic, fn_magic, tp_m = confusion_matrix(y_true, results_df['pred_magic'], labels=[0, 1]).ravel()
    
    print("\n" + "="*40)
    print("TEST SET ANALYSIS")
    print("="*40)
    print(f"Total Images: {len(results_df)}")
    print(f"Magic Tool Triggers: {magic_triggers} ({magic_triggers/len(results_df)*100:.1f}%)")
    print("-" * 40)
    print(f"STANDARD ENSEMBLE:")
    print(f"  Balanced Accuracy: {ba_std:.4f}")
    print(f"  False Positives:   {fp_std}")
    print(f"  False Negatives:   {fn_std}")
    print("-" * 40)
    print(f"WITH MAGIC TOOL (Always AI if triggered):")
    print(f"  Balanced Accuracy: {ba_magic:.4f}")
    print(f"  False Positives:   {fp_magic}")
    print(f"  False Negatives:   {fn_magic}")
    print("="*40)
    
    if ba_magic > ba_std:
        print(f"🚀 Magic Tool IMPROVED accuracy by {ba_magic - ba_std:.4f}")
    else:
        print(f"📊 Magic Tool impact: {ba_magic - ba_std:.4f}")

if __name__ == "__main__":
    main()
