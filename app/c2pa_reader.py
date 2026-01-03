import json
import c2pa
from typing import Optional, Dict, Any

def get_c2pa_manifest(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Reads and validates C2PA manifest data from a media file.
    Returns the active manifest or the first manifest found.
    """
    try:
        with c2pa.Reader(file_path) as reader:
            json_data = reader.json()
            manifest_store = json.loads(json_data)
            
            # DEBUG LOG for production - helps identify Gemini manifest structure
            print(f"C2PA Manifest Store Keys: {list(manifest_store.keys())}")
            if "active_manifest" in manifest_store:
                print(f"Active Manifest Label: {manifest_store['active_manifest']}")
            if active_label and active_label in manifest_store.get("manifests", {}):
                return manifest_store["manifests"][active_label]
            
            # 2. Fallback: If no active manifest, check any manifests present
            manifests = manifest_store.get("manifests", {})
            if manifests:
                # Return the first available manifest
                first_label = next(iter(manifests))
                return manifests[first_label]
                
    except Exception as e:
        print(f"C2PA Reader Error: {e}")
        pass
    return None
