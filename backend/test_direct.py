import os
import io
import torch
from PIL import Image
from app.services.authenticity_service import authenticity_service
from app.models.model_loader import model_loader

def test_direct():
    print("Direct Test: Loading models...")
    model_loader.load_all_models()
    
    # Create dummy image
    img = Image.new('RGB', (224, 224), color=(73, 109, 137))
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    print("Direct Test: Calling verify...")
    try:
        res = authenticity_service.verify(img_bytes, "Nike")
        print("Direct Test: SUCCESS!")
        print(f"Verdict: {res.verdict}, Conf: {res.confidence}%")
        print(f"Heatmap Length: {len(res.heatmap_base64)}")
    except Exception as e:
        print(f"Direct Test: FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct()
