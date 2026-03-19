import os
import requests
import base64
import json
from tqdm import tqdm

# Configuration
API_URL = "http://127.0.0.1:8000/predict/"
DATASET_DIR = "../dataset/processed"
OUTPUT_DIR = "test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def test_single(image_path, brand_claim, label):
    """Tests a single image against a brand claim and prints results."""
    if not os.path.exists(image_path):
        print(f"Skipping {image_path} (not found)")
        return None

    try:
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            data = {"brand": brand_claim}
            r = requests.post(API_URL, files=files, data=data, timeout=30)
            r.raise_for_status()
            res = r.json()
    except Exception as e:
        print(f"Error testing {image_path}: {e}")
        return None

    # Step 8 Requirement: Asserts authentic >= 0.7, fake <= 0.3
    is_authentic = res["verdict"] == "authentic"
    conf = res["confidence"] / 100.0
    
    status = "PASS"
    if label == "authentic" and (not is_authentic or conf < 0.7):
        status = "FAIL (Authentic shoe scored low)"
    elif label == "fake" and (is_authentic or (1.0 - conf) < 0.7):
        # Here "fake" means we expect a low authentic probability
        status = "FAIL (Fake shoe/mismatch scored high)"

    print(f"[{status}] {os.path.basename(image_path)} -> Label: {label}, Verdict: {res['verdict']}, Conf: {res['confidence']}%, Level: {res['confidence_level']}")

    # Save heatmap for visual verification (Step 8)
    if res.get("heatmap_base64"):
        h_data = res["heatmap_base64"].split(",")[1]
        out_name = f"{label}_{brand_claim}_{os.path.basename(image_path)}"
        with open(os.path.join(OUTPUT_DIR, out_name), "wb") as f:
            f.write(base64.b64decode(h_data))
            
    return res

def run_e2e_tests():
    print("Starting Step 8: End-to-End Inference Testing...")
    
    # 1. Pick 3 Authentic Images (Nike, Adidas, Gucci)
    auth_cases = [
        ("Nike", os.path.join(DATASET_DIR, "Nike", "7172014.12309.jpg")),
        ("Adidas", os.path.join(DATASET_DIR, "Adidas", os.listdir(os.path.join(DATASET_DIR, "Adidas"))[0])),
        ("Gucci", os.path.join(DATASET_DIR, "Gucci", os.listdir(os.path.join(DATASET_DIR, "Gucci"))[0])),
    ]

    # 2. Pick 3 "Fake" (Mismatch) Images
    # We simulate fakes by claiming a different brand or using random noise
    fake_cases = [
        ("Nike", os.path.join(DATASET_DIR, "Adidas", os.listdir(os.path.join(DATASET_DIR, "Adidas"))[1])),
        ("Adidas", os.path.join(DATASET_DIR, "Nike", os.listdir(os.path.join(DATASET_DIR, "Nike"))[1])),
        ("Gucci", os.path.join(DATASET_DIR, "Vans", os.listdir(os.path.join(DATASET_DIR, "Vans"))[0])),
    ]

    print("\n--- Testing Authentic Shoes ---")
    for brand, path in auth_cases:
        test_single(path, brand, "authentic")

    print("\n--- Testing 'Fake' (Brand Mismatch) Shoes ---")
    for brand, path in fake_cases:
        test_single(path, brand, "fake")

    print(f"\nHeatmaps saved to ./{OUTPUT_DIR}/ for visual audit.")

if __name__ == "__main__":
    # Ensure uvicorn is running first!
    try:
        requests.get("http://127.0.0.1:8000/health/", timeout=2)
        run_e2e_tests()
    except Exception as e:
        print(f"ERROR: Backend is not running on http://127.0.0.1:8000/health/ ({e})")
        print("Please start uvicorn: python -m uvicorn app.main:app --reload")
