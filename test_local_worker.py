import requests
import base64
import time
import sys
import json
from PIL import Image
import io

# Dummy base64 image (small white square)
def get_dummy_b64():
    img = Image.new('RGB', (224, 224), color='white')
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def test_worker_local(port=8000):
    url = f"http://localhost:{port}/run"
    payload = {
        "input": {
            "image": get_dummy_b64()
        }
    }
    
    print(f"🚀 Sending test request to local worker at {url}...")
    t0 = time.time()
    try:
        # RunPod local test server uses /run or is just the handler
        # If running via 'python handler.py' locally, it might not have a web server 
        # unless started via runpod.serverless.start() with a local mock.
        # However, we can simulate the handler call directly if we import it.
        print("💡 Tip: To test without Docker, run: python worker/handler.py --test")
        
        # We'll assume the user might be running the docker container mapping port 8000
        response = requests.post(url, json=payload, timeout=60)
        print(f"✅ Status: {response.status_code}")
        print(f"📦 Response: {json.dumps(response.json(), indent=2)}")
        print(f"⏱️  Total Time: {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"❌ Failed to reach worker: {e}")

if __name__ == "__main__":
    test_worker_local()
