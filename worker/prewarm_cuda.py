import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prewarm")

def prewarm():
    if torch.cuda.is_available():
        logger.info("🔥 Warming up CUDA context...")
        torch.randn(1, device="cuda")
        logger.info("✨ CUDA context initialized.")
    else:
        logger.info("⚠️ CUDA not available during build pre-warm.")

if __name__ == "__main__":
    prewarm()

