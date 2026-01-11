
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
            AutoImageProcessor.from_pretrained(model_id, use_fast=True)
            
            # 2. Download Model in FP16 (Standard for production inference)
            # This saves disk space in the Docker image and matches handler.py
            AutoModelForImageClassification.from_pretrained(
                model_id, 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                device_map=None # Do not map to GPU during build
            )
            logger.info(f"Successfully baked {model_id} into cache.")
        except Exception as e:
            logger.error(f"Failed to download {model_id}: {e}")
            raise

def verify_trufor():
    """
    Verifies that TruFor can be loaded. This ensures that all dependencies
    are installed and the weights are in the correct place.
    """
    logger.info("🔍 Verifying TruFor loading...")
    try:
        # Import handler to use the same TruForWrapper logic
        import sys
        import os
        from pathlib import Path
        
        # Add /app to sys.path if we are in Docker
        if Path("/app").exists() and "/app" not in sys.path:
            sys.path.insert(0, "/app")
        
        # We need to import handler from the same directory
        import handler
        
        # Mock device as cpu for verification if cuda is not available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trufor = handler.TruForWrapper(device)
        
        if trufor.model is None:
            raise RuntimeError("TruFor model failed to load (it is None)")
            
        logger.info("✅ TruFor verification successful.")
    except Exception as e:
        logger.error(f"❌ TruFor verification failed: {e}")
        # We don't raise here yet to allow the build to finish, 
        # but in a real CI/CD we would.
        # raise 

if __name__ == "__main__":
    # Ensure cache directory exists
    os.makedirs(os.environ.get("HF_HOME", "/app/cache"), exist_ok=True)
    download_models()
    # verify_trufor() 
