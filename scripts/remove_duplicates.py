import os
import hashlib
from tqdm import tqdm

def get_file_hash(filepath):
    """Calculate MD5 hash of file content"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def remove_duplicates(target_dir):
    print(f"\n--- Cleaning duplicates in {target_dir} ---")
    if not os.path.exists(target_dir):
        print(f"⚠️ Directory {target_dir} does not exist.")
        return 0
    
    hashes = {}
    files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    removed_count = 0
    
    for filename in tqdm(files, desc=f"Hashing {os.path.basename(target_dir)}"):
        filepath = os.path.join(target_dir, filename)
        file_hash = get_file_hash(filepath)
        
        if file_hash in hashes:
            # Duplicate found
            # print(f"Removing duplicate: {filename} (same as {hashes[file_hash]})")
            os.remove(filepath)
            removed_count += 1
        else:
            hashes[file_hash] = filename
            
    return removed_count

if __name__ == "__main__":
    BASE_DIR = "/Users/netanel.ossi/Desktop/ai_detector_dataset"
    AI_DIR = os.path.join(BASE_DIR, "ai")
    REAL_DIR = os.path.join(BASE_DIR, "real")
    
    total_removed = 0
    total_removed += remove_duplicates(AI_DIR)
    total_removed += remove_duplicates(REAL_DIR)
    
    print(f"\n🎉 Cleanup Complete!")
    print(f"   Total duplicates removed: {total_removed}")
    
    # Final counts
    num_ai = len([f for f in os.listdir(AI_DIR) if f.endswith('.jpg')])
    num_real = len([f for f in os.listdir(REAL_DIR) if f.endswith('.jpg')])
    print(f"   Remaining AI Images: {num_ai}")
    print(f"   Remaining Real Images: {num_real}")

