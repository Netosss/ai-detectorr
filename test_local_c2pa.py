import os
import sys
import json

# Ensure we can import from 'app'
sys.path.append(os.getcwd())

from app.c2pa_reader import get_c2pa_manifest

def test_manifest_extraction():
    test_image = "third_party/c2pa-python/tests/fixtures/files-for-reading-tests/CA.jpg"
    if not os.path.exists(test_image):
        print(f"Error: {test_image} not found.")
        return

    print(f"Testing manifest extraction for: {test_image}")
    manifest = get_c2pa_manifest(test_image)
    
    if manifest:
        print("SUCCESS: Manifest found!")
        print(f"Generator: {manifest.get('claim_generator')}")
        print(f"Generator Info: {manifest.get('claim_generator_info')}")
    else:
        print("FAILURE: No manifest found.")

if __name__ == "__main__":
    test_manifest_extraction()

