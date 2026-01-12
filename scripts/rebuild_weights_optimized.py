import os
import json
import time
import torch
import logging
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForImageClassification
from tqdm import tqdm
import optuna
from typing import Dict, List, Any, Optional
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from PIL.ExifTags import TAGS

# --- Metadata Scoring (Copied from app/detectors/core.py for standalone use) ---
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

def get_ai_suspicion_score(exif: dict, width: int = 0, height: int = 0, file_size: int = 0) -> float:
    score = 0.0
    has_camera_info = exif.get("Make") or exif.get("Model")
    ai_keywords = ["stable", "diffusion", "midjourney", "dalle", "flux", "sora", "generative"]
    software = str(exif.get("Software", "")).lower()
    make = str(exif.get("Make", "")).lower()
    if any(k in software for k in ai_keywords) or any(k in make for k in ai_keywords): score += 0.40
    if not has_camera_info: score += 0.10
    if width > 0 and height > 0 and not has_camera_info:
        if width in [512, 768, 1024, 1536, 2048] or height in [512, 768, 1024, 1536, 2048]:
            score += 0.15
    return round(min(score, 1.0), 2)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration & Constants ---
MODELS_TO_TEST = [
    "Ateeqq/ai-vs-human-image-detector",
    "haywoodsloan/ai-image-detector-dev-deploy",
    "Organika/sdxl-detector"
]

# Slice Logic (Mirroring app/detectors/core.py with priority fix)
def get_slice_name(pixels, width, height, ext, aspect):
    if pixels < 2000: return "thumbnail"
    if 2000 <= pixels < 10000: return "low_res"
    if 10000 <= pixels < 50000: return "10k-50k"
    if 50000 <= pixels < 500000: return "50k-500k"
    if pixels >= 500000: return ">500k"
    if ext == ".png": return "png"
    if aspect < 0.8: return "portrait_tall"
    if 0.8 <= aspect <= 1.2: return "squareish"
    return "default"

def logit(p):
    p = max(min(p, 0.999999), 0.000001)
    return np.log(p / (1.0 - p))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --- Pre-processing Helpers ---
def apply_sharpen(img: Image.Image, percent: int) -> Image.Image:
    return img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=percent, threshold=0))

def apply_upscale(img: Image.Image, size: int) -> Image.Image:
    # size is the target height/width (square assumed or max dimension)
    w, h = img.size
    if w >= h:
        new_w = size
        new_h = int(h * (size / w))
    else:
        new_h = size
        new_w = int(w * (size / h))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

# --- Model Wrapper ---
class AIModel:
    def __init__(self, model_id: str, device: str = "cpu"):
        self.model_id = model_id
        self.device = device
        
        # MPS/M2 optimization: half precision can be tricky on some MPS ops
        # but float16 is generally supported on M2 for most layers
        self.dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
        
        logger.info(f"Loading model: {model_id} on {device}")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        
        # Load model
        self.model = AutoModelForImageClassification.from_pretrained(
            model_id, torch_dtype=self.dtype
        ).to(self.device).eval()
        
        # Determine native input size
        self.native_size = 224 # Default
        if hasattr(self.processor, "size"):
            if isinstance(self.processor.size, dict):
                self.native_size = self.processor.size.get("shortest_edge", 224)
            else:
                self.native_size = getattr(self.processor.size, "width", 224)
        
        # Find AI label index
        self.ai_idx = 0
        for idx, label in self.model.config.id2label.items():
            if any(k in label.lower() for k in ['ai', 'fake', 'generated']):
                self.ai_idx = idx
                break
                
        # Optimization: torch.compile is primarily for CUDA right now
        if hasattr(torch, "compile") and device == "cuda":
            try:
                self.model = torch.compile(self.model)
                logger.info(f"Successfully compiled {model_id}")
            except Exception as e:
                logger.warning(f"Could not compile {model_id}: {e}")

    @torch.no_grad()
    def predict(self, images: List[Image.Image]) -> np.ndarray:
        if not images: return np.array([])
        
        # Batching for performance (especially on M2)
        batch_size = 32 if self.device == "mps" else 16
        all_probs = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            
            # Cast inputs to model's dtype
            if self.device in ["cuda", "mps"]:
                inputs = {k: v.to(self.dtype) if v.is_floating_point() else v for k, v in inputs.items()}
            else:
                inputs = {k: v.to(torch.float32) if v.is_floating_point() else v for k, v in inputs.items()}
                
            try:
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_probs.append(probs[:, self.ai_idx].cpu().numpy())
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                # Fallback: if MPS fails with float16, you might need to cast to float32
                all_probs.append(np.zeros(len(batch)))
            
        return np.concatenate(all_probs)

# --- Optimization Orchestrator ---
class OptimizerOrchestrator:
    def __init__(self, data_csvs: List[str], dry_run: bool = False):
        self.dry_run = dry_run
        
        # --- Mac M2 / MPS Optimization ---
        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Using Mac M2 GPU (MPS acceleration)")
        elif torch.cuda.is_available():
            self.device = "cuda"
            logger.info("Using NVIDIA GPU (CUDA acceleration)")
        else:
            self.device = "cpu"
            logger.info("Using CPU")
            
        self.df = self.load_data(data_csvs)
        self.models: Dict[str, AIModel] = {}
        self.best_preprocess: Dict[str, Dict] = {}
        self.phase1_df: Optional[pd.DataFrame] = None
        
    def load_data(self, csv_paths: List[str]) -> pd.DataFrame:
        dfs = []
        for path in csv_paths:
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))
        
        if not dfs:
            raise FileNotFoundError("No valid data CSVs found!")
            
        full_df = pd.concat(dfs).reset_index(drop=True)
        if self.dry_run:
            # Take 4 from each label if available
            ai_subset = full_df[full_df['label_true'] == 'ai'].head(4)
            real_subset = full_df[full_df['label_true'] == 'real'].head(4)
            full_df = pd.concat([ai_subset, real_subset]).reset_index(drop=True)
            logger.info(f"Dry run: Using {len(full_df)} images (4 AI, 4 Real)")
        
        # Add metadata features
        full_df['ext'] = full_df['path'].apply(lambda x: Path(x).suffix.lower())
        
        def get_dims(path):
            try:
                if not os.path.exists(path): return (224, 224)
                with Image.open(path) as img:
                    return img.size
            except:
                return (224, 224) # Fallback to standard square
        
        tqdm.pandas(desc="Loading image metadata")
        dims = full_df['path'].progress_apply(get_dims)
        full_df['width'] = [d[0] for d in dims]
        full_df['height'] = [d[1] for d in dims]
        full_df['pixels'] = full_df['width'] * full_df['height']
        full_df['aspect'] = full_df['width'] / full_df['height']
        full_df['slice'] = full_df.apply(lambda row: get_slice_name(row['pixels'], row['width'], row['height'], row['ext'], row['aspect']), axis=1)
        full_df['label_binary'] = full_df['label_true'].apply(lambda x: 1 if x == 'ai' or x == 'fake' else 0)
        
        # Pre-calculate Metadata Scores
        logger.info("Calculating metadata scores...")
        tqdm.pandas(desc="Metadata Scoring")
        def calc_meta(path, w, h):
            exif = get_exif_data(path)
            f_size = os.path.getsize(path) if os.path.exists(path) else 0
            m_h = get_forensic_metadata_score(exif)
            m_ai = get_ai_suspicion_score(exif, w, h, f_size)
            return pd.Series([m_h, m_ai])
        
        full_df[['m_h', 'm_ai']] = full_df.progress_apply(lambda r: calc_meta(r['path'], r['width'], r['height']), axis=1)
        
        return full_df

    def run_phase_1(self):
        logger.info("=== Phase 1: Pre-processing Optimization (Subset Optimized) ===")
        results_file = "phase1_results.jsonl"
        all_phase1_data = []

        # 1. Create a random subset for the grid search (1k AI, 1k Real)
        if not self.dry_run:
            ai_indices = self.df[self.df['label_binary'] == 1].index
            real_indices = self.df[self.df['label_binary'] == 0].index
            
            # Cap at what's available
            n_ai = min(1000, len(ai_indices))
            n_real = min(1000, len(real_indices))
            
            subset_indices = np.concatenate([
                np.random.choice(ai_indices, n_ai, replace=False),
                np.random.choice(real_indices, n_real, replace=False)
            ])
            subset_df = self.df.loc[subset_indices].copy()
            logger.info(f"Phase 1 Optimization subset: {len(subset_df)} images ({n_ai} AI, {n_real} Real)")
        else:
            subset_df = self.df # Use the small dry run set as is

        for model_id in MODELS_TO_TEST:
            if model_id not in self.models:
                self.models[model_id] = AIModel(model_id, self.device)
            
            model = self.models[model_id]
            
            sharpen_values = [100, 110, 120, 130, 140, 150, 160]
            upscale_values = [None, 224, 448, 512]
            
            best_acc = -1
            best_config = {"sharpen": 100, "upscale": None}
            capped_upscale_limit = min(2 * model.native_size, 512)
            
            # --- STEP A: Grid Search on SUBSET ---
            logger.info(f"Running pre-processing grid search for {model_id} on subset...")
            for up in upscale_values:
                if up is not None:
                    if up > capped_upscale_limit or up < model.native_size:
                        continue
                
                for sh in sharpen_values:
                    config_id = f"up_{up}_sh_{sh}"
                    
                    all_scores_subset = []
                    batch_size = 32
                    for i in range(0, len(subset_df), batch_size):
                        batch_rows = subset_df.iloc[i:i+batch_size]
                        batch_imgs = []
                        for _, row in batch_rows.iterrows():
                            img = Image.open(row['path']).convert("RGB")
                            if up: img = apply_upscale(img, up)
                            if sh > 100: img = apply_sharpen(img, sh)
                            batch_imgs.append(img)
                        
                        scores = model.predict(batch_imgs)
                        all_scores_subset.extend(scores.tolist())
                    
                    preds_binary = [1 if p > 0.5 else 0 for p in all_scores_subset]
                    acc = (np.array(preds_binary) == subset_df['label_binary'].values).mean()
                    
                    if acc > best_acc:
                        best_acc = acc
                        best_config = {"sharpen": sh, "upscale": up}

            self.best_preprocess[model_id] = best_config
            logger.info(f"BEST CONFIG for {model_id}: {best_config} (Subset Acc: {best_acc:.4f})")

            # --- STEP B: Full Inference with BEST CONFIG ---
            logger.info(f"Applying best config to FULL dataset for {model_id}...")
            best_up = best_config["upscale"]
            best_sh = best_config["sharpen"]
            
            full_scores = []
            batch_size = 32
            for i in tqdm(range(0, len(self.df), batch_size), desc=f"{model_id} - Full Inference"):
                batch_rows = self.df.iloc[i:i+batch_size]
                batch_imgs = []
                for _, row in batch_rows.iterrows():
                    img = Image.open(row['path']).convert("RGB")
                    if best_up: img = apply_upscale(img, best_up)
                    if best_sh > 100: img = apply_sharpen(img, best_sh)
                    batch_imgs.append(img)
                
                scores = model.predict(batch_imgs)
                full_scores.extend(scores.tolist())

            # Store full dataset results for Phase 2
            for idx, score in enumerate(full_scores):
                row = self.df.iloc[idx]
                all_phase1_data.append({
                    "path": row['path'],
                    "model": model_id,
                    "upscale": best_up,
                    "sharpen": best_sh,
                    "score": float(score),
                    "label": int(row['label_binary']),
                    "m_h": float(row['m_h']),
                    "m_ai": float(row['m_ai'])
                })

        with open(results_file, 'w') as f:
            for entry in all_phase1_data:
                f.write(json.dumps(entry) + '\n')
        
        self.phase1_df = pd.DataFrame(all_phase1_data)
        logger.info(f"Phase 1 complete. Best config inference results saved for all images.")

    def run_phase_2(self):
        logger.info("=== Phase 2: Ensemble Weight Optimization (Optuna) ===")
        if self.phase1_df is None:
            self.phase1_df = pd.read_json("phase1_results.jsonl", lines=True)

        combinations = [
            [MODELS_TO_TEST[0]], [MODELS_TO_TEST[1]], [MODELS_TO_TEST[2]],
            [MODELS_TO_TEST[0], MODELS_TO_TEST[1]],
            [MODELS_TO_TEST[0], MODELS_TO_TEST[2]],
            [MODELS_TO_TEST[1], MODELS_TO_TEST[2]],
            MODELS_TO_TEST
        ]

        ensemble_results = []

        for combo in combinations:
            logger.info(f"Optimizing ensemble for combination: {combo}")
            
            # Prepare data for this combo using best pre-processing found in Phase 1
            combo_data = {}
            for model_id in combo:
                best = self.best_preprocess.get(model_id, {"sharpen": 100, "upscale": None})
                mask = (self.phase1_df['model'] == model_id) & \
                       (self.phase1_df['sharpen'] == best['sharpen']) & \
                       (self.phase1_df['upscale'].fillna(0) == (best['upscale'] if best['upscale'] is not None else 0))
                
                # Align with self.df using path
                m_df = self.phase1_df[mask].set_index('path')
                combo_data[model_id] = m_df.reindex(self.df['path'])['score'].apply(logit).values

            # Metadata for gating
            m_h_all = self.df['m_h'].values
            m_ai_all = self.df['m_ai'].values

            # Optimize per slice
            slices = self.df['slice'].unique()
            combo_config = {"combination": combo, "slices": {}}

            for slice_name in slices:
                logger.info(f"  Optimizing slice: {slice_name}")
                slice_mask = self.df['slice'] == slice_name
                y_true = self.df[slice_mask]['label_binary'].values
                m_h_slice = m_h_all[slice_mask]
                m_ai_slice = m_ai_all[slice_mask]
                
                if len(y_true) == 0: continue
                
                slice_logits = {m: scores[slice_mask] for m, scores in combo_data.items()}
                
                # Optuna Study
                def objective(trial):
                    # Unbiased Normalization Method
                    p = []
                    for i in range(len(combo)):
                        p.append(trial.suggest_float(f"p{i}", 0, 1))
                    
                    sum_p = sum(p) if sum(p) > 0 else 1.0
                    weights = [val / sum_p for val in p]
                    
                    alpha = trial.suggest_float("alpha", 0.0, 1.0)
                    tau = trial.suggest_float("tau", 0.0, 1.0) # Larger range for logit-space tau
                    margin = trial.suggest_float("margin", -2.0, 2.0)
                    
                    # Ensemble Logic
                    l_total = np.zeros_like(y_true, dtype=np.float64)
                    for i, model_id in enumerate(combo):
                        l_total += weights[i] * slice_logits[model_id]
                    
                    # Metadata Gating (Tau)
                    # Implementation from app/detectors/core.py:
                    # if abs(l_total) < p['tau']:
                    #     meta_signal = m_ai - m_h
                    #     l_final = (p['alpha'] * l_total) + ((1 - p['alpha']) * meta_signal)
                    
                    l_final = np.copy(l_total)
                    gate_mask = np.abs(l_total) < tau
                    meta_signal = m_ai_slice - m_h_slice
                    l_final[gate_mask] = (alpha * l_total[gate_mask]) + ((1 - alpha) * meta_signal[gate_mask])
                    
                    preds = (l_final > margin).astype(int)
                    
                    try:
                        return balanced_accuracy_score(y_true, preds)
                    except:
                        return 0.0

                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=10 if self.dry_run else 100)
                
                best_params = study.best_params
                sum_p = sum(best_params[f"p{i}"] for i in range(len(combo)))
                final_weights = {combo[i]: best_params[f"p{i}"] / sum_p for i in range(len(combo))}
                
                combo_config["slices"][slice_name] = {
                    "weights": final_weights,
                    "alpha": best_params["alpha"],
                    "tau": best_params["tau"],
                    "margin": best_params["margin"],
                    "best_ba": study.best_value
                }

            ensemble_results.append(combo_config)

        self.ensemble_results = ensemble_results
        with open("ensemble_configs.json", 'w') as f:
            json.dump(ensemble_results, f, indent=4)
        logger.info("Phase 2 results saved to ensemble_configs.json")

    def run_phase_3_4(self):
        logger.info("=== Phase 3 & 4: Threshold & Suspicion Tuning ===")
        # Pick best ensemble
        best_combo_idx = 0
        best_avg_ba = -1
        for i, res in enumerate(self.ensemble_results):
            avg_ba = np.mean([s["best_ba"] for s in res["slices"].values()])
            if avg_ba > best_avg_ba:
                best_avg_ba = avg_ba
                best_combo_idx = i
        
        self.winner = self.ensemble_results[best_combo_idx]
        logger.info(f"Winning combination: {self.winner['combination']} (Avg BA: {best_avg_ba:.4f})")
        
        # Pull final l_final for each image using winning configs
        combo = self.winner["combination"]
        l_finals = np.zeros(len(self.df))
        
        # Need to reconstruct combo_data as in run_phase_2
        combo_data = {}
        for model_id in combo:
            best = self.best_preprocess.get(model_id, {"sharpen": 100, "upscale": None})
            mask = (self.phase1_df['model'] == model_id) & \
                   (self.phase1_df['sharpen'] == best['sharpen']) & \
                   (self.phase1_df['upscale'].fillna(0) == (best['upscale'] if best['upscale'] is not None else 0))
            m_df = self.phase1_df[mask].set_index('path')
            combo_data[model_id] = m_df.reindex(self.df['path'])['score'].apply(logit).values

        m_h_all = self.df['m_h'].values
        m_ai_all = self.df['m_ai'].values

        for slice_name, cfg in self.winner["slices"].items():
            slice_mask = self.df['slice'] == slice_name
            if not any(slice_mask): continue
            
            l_total = np.zeros(sum(slice_mask))
            for i, model_id in enumerate(combo):
                l_total += cfg['weights'][model_id] * combo_data[model_id][slice_mask]
            
            l_final_slice = np.copy(l_total)
            gate_mask = np.abs(l_total) < cfg['tau']
            meta_signal = m_ai_all[slice_mask] - m_h_all[slice_mask]
            l_final_slice[gate_mask] = (cfg['alpha'] * l_total[gate_mask]) + ((1 - cfg['alpha']) * meta_signal[gate_mask])
            l_finals[slice_mask] = l_final_slice

        # Now optimize global thresholds and suspicion window
        y_true = self.df['label_binary'].values
        
        def objective(trial):
            h_th = trial.suggest_float("h_th", 0.1, 0.9)
            ai_th = trial.suggest_float("ai_th", 0.1, 0.9)
            susp_win = trial.suggest_float("susp_win", 0.05, 0.5)
            
            # Simulated effect: if metadata exceeds thresholds, it bypasses ensemble
            # (In reality this happens BEFORE GPU scan, but here we tune for best overall)
            final_preds = np.zeros_like(y_true)
            for i in range(len(self.df)):
                if m_h_all[i] > h_th: final_preds[i] = 0 # Verified Human
                elif m_ai_all[i] > ai_th: final_preds[i] = 1 # Verified AI
                else:
                    # Use ensemble result
                    slice_name = self.df.iloc[i]['slice']
                    margin = self.winner["slices"][slice_name]["margin"]
                    final_preds[i] = 1 if l_finals[i] > margin else 0
            
            # We want to maximize BA while keeping suspicion useful
            try:
                return balanced_accuracy_score(y_true, final_preds)
            except:
                return 0.0

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10 if self.dry_run else 100)
        
        self.winner["dual_gate_metadata"] = {
            "human_threshold": study.best_params["h_th"],
            "ai_threshold": study.best_params["ai_th"]
        }
        self.winner["suspicion"] = {
            "window": study.best_params["susp_win"],
            "center": "margin"
        }
        logger.info(f"Optimized thresholds: {self.winner['dual_gate_metadata']}")
        logger.info(f"Optimized suspicion window: {self.winner['suspicion']['window']}")

    def generate_report(self):
        logger.info("=== Phase 5: Reporting & Export ===")
        report_path = "reports/ensemble_comparison_report.md"
        os.makedirs("reports", exist_ok=True)
        
        # Calculate Final Stats for Winning Ensemble
        combo = self.winner["combination"]
        combo_data = {}
        for model_id in combo:
            best = self.best_preprocess.get(model_id, {"sharpen": 100, "upscale": None})
            mask = (self.phase1_df['model'] == model_id) & \
                   (self.phase1_df['sharpen'] == best['sharpen']) & \
                   (self.phase1_df['upscale'].fillna(0) == (best['upscale'] if best['upscale'] is not None else 0))
            m_df = self.phase1_df[mask].set_index('path')
            combo_data[model_id] = m_df.reindex(self.df['path'])['score'].apply(logit).values

        with open(report_path, "w") as f:
            f.write("# Automated Ensemble Optimization Report\n\n")
            f.write(f"Generated at: {time.ctime()}\n\n")
            
            f.write("## Winning Configuration\n")
            f.write(f"**Models**: {', '.join(combo)}\n\n")
            
            f.write("### Slice-Wise Performance & Weights\n")
            f.write("| Slice | BA | Margin | FP | FN | w1 | w2 | w3 |\n")
            f.write("|-------|----|--------|----|----|----|----|----|\n")
            
            suspicion_log = []

            for slice_name, cfg in self.winner["slices"].items():
                slice_mask = self.df['slice'] == slice_name
                y_true = self.df[slice_mask]['label_binary'].values
                if len(y_true) == 0: continue
                
                # Reconstruct predictions for report
                l_total = np.zeros_like(y_true, dtype=np.float64)
                for i, model_id in enumerate(combo):
                    l_total += cfg['weights'][model_id] * combo_data[model_id][slice_mask]
                
                preds = (l_total > cfg['margin']).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
                
                # Tracking suspicion
                susp_mask = np.abs(l_total - cfg['margin']) < self.winner["suspicion"]["window"]
                for path in self.df[slice_mask][susp_mask]['path']:
                    suspicion_log.append(path)

                w_list = [f"{cfg['weights'].get(m, 0.0):.3f}" for m in combo]
                while len(w_list) < 3: w_list.append("-")
                w_str = " | ".join(w_list)
                
                f.write(f"| {slice_name} | {cfg['best_ba']:.4f} | {cfg['margin']:.2f} | {fp} | {fn} | {w_str} |\n")
            
            f.write("\n## Model Contribution Analysis\n")
            for model_id in combo:
                avg_w = np.mean([s["weights"].get(model_id, 0.0) for s in self.winner["slices"].values()])
                f.write(f"- **{model_id}**: Average weight contribution: {avg_w:.2f}\n")

            f.write("\n## Suspicion Window Tracking\n")
            f.write(f"Found {len(suspicion_log)} images in the suspicion window (|logit - margin| < {self.winner['suspicion']['window']}):\n\n")
            for path in suspicion_log[:50]: # Cap log for readability
                f.write(f"- `{path}`\n")
            if len(suspicion_log) > 50:
                f.write(f"- ... and {len(suspicion_log)-50} more.\n")

        # Export Production Config
        prod_config = {
            "dual_gate_metadata": self.winner["dual_gate_metadata"],
            "slices": self.winner["slices"],
            "suspicion": self.winner["suspicion"]
        }
        with open("configs/new_optimized_config.json", "w") as f:
            json.dump(prod_config, f, indent=4)
        
        logger.info(f"Report generated: {report_path}")
        logger.info("Production config saved: configs/new_optimized_config.json")

def main():
    parser = argparse.ArgumentParser(description="Automated AI Detector Optimizer")
    parser.add_argument("--dry-run", action="store_true", help="Run with minimal data for verification")
    args = parser.parse_args()

    data_files = [
        "data/training_dataset.csv"
    ]
    
    orchestrator = OptimizerOrchestrator(data_files, dry_run=args.dry_run)
    orchestrator.run_phase_1()
    orchestrator.run_phase_2()
    orchestrator.run_phase_3_4()
    orchestrator.generate_report()

if __name__ == "__main__":
    main()

