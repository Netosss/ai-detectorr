from app.c2pa_reader import get_c2pa_manifest

async def detect_ai_media(file_path: str) -> dict:
    """
    Detect if an image/video is AI-generated based on C2PA metadata.
    This is a deterministic provenance check.
    """
    manifest = get_c2pa_manifest(file_path)
    
    if manifest:
        # 1. Try to get specific product name from generator info
        gen_info = manifest.get("claim_generator_info", [])
        generator = "Unknown AI"
        
        if gen_info and isinstance(gen_info, list):
            generator = gen_info[0].get("name", generator)
        else:
            generator = manifest.get("claim_generator", generator)

        # 2. Check assertions for explicit AI labels (e.g. 'c2pa.ai_generated')
        is_explicitly_ai = False
        assertions = manifest.get("assertions", [])
        for assertion in assertions:
            label = assertion.get("label", "")
            if "ai_generated" in label.lower() or "ai_inference" in label.lower():
                is_explicitly_ai = True
                break

        return {
            "is_ai": True,
            "provider": generator,
            "method": "c2pa",
            "confidence": 1.0,
            "explicit_ai_assertion": is_explicitly_ai
        }
    
    # If missing, we cannot claim AI deterministically
    return {
        "is_ai": None,
        "provider": None,
        "method": "none",
        "confidence": 0.0
    }

