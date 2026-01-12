import os
import io
import webdataset as wds
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Configuration
BASE_DIR = "/Users/netanel.ossi/Desktop/ai_detector_dataset"
AI_DIR = os.path.join(BASE_DIR, "ai")
REAL_DIR = os.path.join(BASE_DIR, "real")
MAX_WORKERS = 12

# Ensure directory exists
os.makedirs(AI_DIR, exist_ok=True)

def save_image_worker(args):
    """Worker function for ThreadPoolExecutor"""
    img_bytes, folder, filename = args
    path = os.path.join(folder, filename)
    
    if os.path.exists(path):
        return True

    try:
        # Open from raw bytes to bypass any formatting issues
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert("RGB")
        img.save(path, "JPEG", quality=90, optimize=True)
        return True
    except Exception:
        return False

def download_dalle3_only():
    print("\n--- Downloading DALL-E 3 (ProGamerGov Direct Stream) ---")
    try:
        # Target: 5,000 images total. We already have ~617.
        # Construct URLs for the shards
        base_url = "https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions/resolve/main/data/data-{i:06d}.tar"
        # We'll use more shards to ensure we get enough
        urls = [base_url.format(i=i) for i in range(20)]
        
        # We use .decode() without arguments to get raw bytes
        dataset = wds.WebDataset(urls).shuffle(1000)
        target = 5000
        count = 0
        
        # Get list of existing files to count accurately
        existing = [f for f in os.listdir(AI_DIR) if f.startswith('dalle3_')]
        count = len(existing)
        print(f"   Already have {count} dalle3 images. Target total: {target}")

        if count >= target:
            print("✅ DALL-E 3 target already reached.")
            return

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            tasks = []
            # Iterate through webdataset items
            for item in tqdm(dataset, desc="Streaming DALL-E 3"):
                if count >= target: break
                
                # In WebDataset, item is a dict where keys are file extensions
                # We want the image bytes from 'jpg' or 'png' or 'jpeg'
                img_bytes = item.get('jpg') or item.get('png') or item.get('jpeg')
                
                if img_bytes:
                    filename = f"dalle3_wds_{count}.jpg"
                    tasks.append(executor.submit(save_image_worker, (img_bytes, AI_DIR, filename)))
                    count += 1
            
            # Wait for completion
            for task in tqdm(tasks, desc="Saving Images"):
                task.result()
                
        print(f"✅ Finished DALL-E 3 download. Total in folder now: {count}")
    except Exception as e:
        print(f"⚠️ DALL-E 3 error: {e}")

if __name__ == "__main__":
    download_dalle3_only()
    
    # Final Stats
    num_ai = len([f for f in os.listdir(AI_DIR) if f.endswith('.jpg')])
    num_real = len([f for f in os.listdir(REAL_DIR) if f.endswith('.jpg')])
    print(f"\n🎉 Progress Update!")
    print(f"   Total AI Images: {num_ai}")
    print(f"   Total Real Images: {num_real}")
