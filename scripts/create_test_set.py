import os
import random
import shutil
from tqdm import tqdm

def create_test_set(base_dir, test_dir, ai_count=1500, real_count=1200):
    print(f"\n--- Creating Test Set in {test_dir} ---")
    
    # Source directories
    ai_src = os.path.join(base_dir, "ai")
    real_src = os.path.join(base_dir, "real")
    
    # Target directories
    ai_test = os.path.join(test_dir, "ai")
    real_test = os.path.join(test_dir, "real")
    
    os.makedirs(ai_test, exist_ok=True)
    os.makedirs(real_test, exist_ok=True)
    
    # Get all available files
    ai_files = [f for f in os.listdir(ai_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    real_files = [f for f in os.listdir(real_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Check if we have enough files
    if len(ai_files) < ai_count:
        print(f"⚠️ Warning: Only {len(ai_files)} AI images available. Taking all.")
        ai_count = len(ai_files)
    if len(real_files) < real_count:
        print(f"⚠️ Warning: Only {len(real_files)} Real images available. Taking all.")
        real_count = len(real_files)
        
    # Select random samples
    ai_sample = random.sample(ai_files, ai_count)
    real_sample = random.sample(real_files, real_count)
    
    # Move files to test set
    print(f"Moving {ai_count} AI images to test folder...")
    for f in tqdm(ai_sample, desc="Moving AI"):
        shutil.move(os.path.join(ai_src, f), os.path.join(ai_test, f))
        
    print(f"Moving {real_count} Real images to test folder...")
    for f in tqdm(real_sample, desc="Moving Real"):
        shutil.move(os.path.join(real_src, f), os.path.join(real_test, f))
        
    print(f"\n✅ Test set created!")
    print(f"   AI test images: {len(os.listdir(ai_test))}")
    print(f"   Real test images: {len(os.listdir(real_test))}")
    
    # Remaining counts in training set
    print(f"\n--- Remaining Training Data ---")
    print(f"   AI: {len(os.listdir(ai_src))}")
    print(f"   Real: {len(os.listdir(real_src))}")

if __name__ == "__main__":
    BASE_DIR = "/Users/netanel.ossi/Desktop/ai_detector_dataset"
    TEST_DIR = "/Users/netanel.ossi/Desktop/ai_detector_test_set"
    
    create_test_set(BASE_DIR, TEST_DIR)

