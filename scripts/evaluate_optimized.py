import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from typing import Dict, List, Any

# --- Configuration ---
CONFIG_PATH = "configs/new_optimized_config.json"
AI_DATASET_PATH = "/Users/netanel.ossi/Desktop/ai-detector-datasets/ai/images"
REAL_DATASET_PATH = "/Users/netanel.ossi/Desktop/ai-detector-datasets/original/images"

# Best pre-processing found in Phase 1
BEST_PREPROCESS = {
    "Ateeqq/ai-vs-human-image-detector": {"sharpen": 110, "upscale": None},
    "haywoodsloan/ai-image-detector-dev-deploy": {"sharpen": 110, "upscale": None},
    "Organika/sdxl-detector": {"sharpen": 100, "upscale": 224}
}

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
            if any(k in label.lower() for k in ['ai', 'fake', 'generated']):
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

def main():
    # 1. Load Config
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    # 2. Setup Device
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Discover Data
    data = []
    for folder, label in [(AI_DATASET_PATH, 1), (REAL_DATASET_PATH, 0)]:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    data.append({'path': os.path.join(folder, f), 'label_true': label})
    
    df = pd.DataFrame(data)
    if df.empty:
        print("No images found in the specified paths!")
        return
    
    print(f"Loaded {len(df)} images for evaluation.")

    # 4. Initialize Models
    model_ids = list(BEST_PREPROCESS.keys())
    models = {mid: AIModel(mid, device) for mid in model_ids}

    # 5. Run Inference
    results = {}
    for mid, model in models.items():
        print(f"Running inference for {mid}...")
        prep = BEST_PREPROCESS[mid]
        all_scores = []
        
        batch_size = 32
        for i in tqdm(range(0, len(df), batch_size)):
            batch_rows = df.iloc[i:i+batch_size]
            batch_imgs = []
            for _, row in batch_rows.iterrows():
                try:
                    img = Image.open(row['path']).convert("RGB")
                    # Calculate pixels for slice lookup later
                    if 'pixels' not in row:
                        df.at[row.name, 'pixels'] = img.size[0] * img.size[1]
                    
                    if prep['upscale']:
                        img = apply_upscale(img, prep['upscale'])
                    if prep['sharpen'] > 100:
                        img = apply_sharpen(img, prep['sharpen'])
                    batch_imgs.append(img)
                except Exception as e:
                    print(f"Error loading {row['path']}: {e}")
                    batch_imgs.append(Image.new('RGB', (224, 224)))
            
            scores = model.predict(batch_imgs)
            all_scores.extend(scores.tolist())
        
        results[mid] = [logit(s) for s in all_scores]

    # 6. Ensemble Calculation
    print("Applying optimized ensemble logic...")
    df['slice'] = df['pixels'].apply(get_slice_name)
    final_preds = []
    
    for idx, row in df.iterrows():
        slice_name = row['slice']
        # Handle case where slice name might not be in config (fallback to default or nearest)
        if slice_name not in config['slices']:
            # Fallback logic
            slice_cfg = list(config['slices'].values())[0] # Just use the first one if missing
        else:
            slice_cfg = config['slices'][slice_name]
            
        l_total = 0
        for mid in model_ids:
            weight = slice_cfg['weights'].get(mid, 0.0)
            l_total += weight * results[mid][idx]
        
        # Note: We are ignoring metadata gating (alpha/tau) here because we don't have
        # the forensic EXIF data readily available for these new files in this simple script.
        # We will use the margin for classification.
        margin = slice_cfg['margin']
        pred = 1 if l_total > margin else 0
        final_preds.append(pred)

    df['pred'] = final_preds
    
    # 7. Results
    y_true = df['label_true'].values
    y_pred = df['pred'].values
    
    ba = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    print("\n" + "="*30)
    print("FINAL EVALUATION RESULTS")
    print("="*30)
    print(f"Overall Balanced Accuracy: {ba:.4f}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print("="*30)

    # Slice-wise breakdown
    print("\nSlice-wise Accuracy:")
    for sn in df['slice'].unique():
        s_df = df[df['slice'] == sn]
        if len(s_df) == 0: continue
        s_ba = balanced_accuracy_score(s_df['label_true'], s_df['pred'])
        print(f"- {sn}: {s_ba:.4f} (n={len(s_df)})")

if __name__ == "__main__":
    main()

