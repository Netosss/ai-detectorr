
import logging
import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Minimal logging for build-time feedback
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("builder")

def download_models():
    """
    Downloads and caches the HuggingFace models used by the worker.
    By baking these into the Docker image, we avoid cold-start delays.
    """
    models = [
        "haywoodsloan/ai-image-detector-dev-deploy", # Model A
        "Ateeqq/ai-vs-human-image-detector"           # Model B
    ]
    
    logger.info("Starting build-time model download...")
    
    for model_id in models:
        logger.info(f"Downloading: {model_id}")
        try:
            # 1. Download Processor
            logger.info(f"  Fetching processor for {model_id}...")
            AutoImageProcessor.from_pretrained(model_id, use_fast=False)
            
            # 2. Download Model in FP16
            logger.info(f"  Fetching model weights for {model_id} (FP16)...")
            AutoModelForImageClassification.from_pretrained(
                model_id, 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=None
            )
            logger.info(f"Successfully baked {model_id} into cache.")
        except Exception as e:
            logger.error(f"Failed to download {model_id}: {e}")
            raise

if __name__ == "__main__":
    # Ensure cache directory exists
    cache_dir = os.environ.get("HF_HOME", "/app/cache")
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"HF_HOME set to: {cache_dir}")
    download_models()
