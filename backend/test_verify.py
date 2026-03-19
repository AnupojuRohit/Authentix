"""
test_verify.py — Quick end-to-end test for the /predict/ endpoint.

Usage (from the backend directory with venv activated):
    python test_verify.py  [image_path]  [brand]

Defaults:
    • image_path = "test_image.jpg"   (create a dummy one if it doesn't exist)
    • brand      = "Nike"

Examples:
    python test_verify.py
    python test_verify.py C:/path/to/shoe.png "Adidas"
    python test_verify.py shoe.webp "Gucci"
"""
import sys
import os
import time
import json
import requests

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL   = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
ENDPOINT   = f"{BASE_URL}/predict/"
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"
BRAND      = sys.argv[2] if len(sys.argv) > 2 else "Nike"
TIMEOUT    = 60  # seconds

# ── Create a tiny dummy JPEG if the user doesn't have a real image ──────────
def ensure_test_image(path: str) -> None:
    if os.path.exists(path):
        return
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (224, 224), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 174, 174], fill=(0, 0, 0))          # black box
        draw.text((70, 100), "TEST", fill=(255, 255, 255))           # text in box
        ext = os.path.splitext(path)[1].lower()
        fmt = {"jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}
        img.save(path, fmt.get(ext, "JPEG"))
        print(f"[test_verify] Created dummy test image: {path}")
    except Exception as e:
        print(f"[test_verify] Could not auto-create test image: {e}")
        print(f"[test_verify] Please put a real image file at: {path}")
        sys.exit(1)

# ── Main test ────────────────────────────────────────────────────────────────
def run_test() -> None:
    print("=" * 60)
    print("  Authentix /predict/ Endpoint Test")
    print("=" * 60)
    print(f"  Endpoint  : {ENDPOINT}")
    print(f"  Brand     : {BRAND}")
    print(f"  Image     : {IMAGE_PATH}")
    print("=" * 60)

    ensure_test_image(IMAGE_PATH)

    with open(IMAGE_PATH, "rb") as img_file:
        mime = "image/jpeg"
        ext = os.path.splitext(IMAGE_PATH)[1].lower()
        if ext == ".png":  mime = "image/png"
        elif ext == ".webp": mime = "image/webp"

        files  = {"image": (os.path.basename(IMAGE_PATH), img_file, mime)}
        data   = {"brand": BRAND}

        start = time.time()
        try:
            resp = requests.post(ENDPOINT, files=files, data=data, timeout=TIMEOUT)
        except requests.exceptions.ConnectionError:
            print("\n[FAIL] Could not connect. Is the backend running?")
            print(f"       uvicorn app.main:app --reload  (from backend/)")
            sys.exit(1)
        except requests.exceptions.Timeout:
            print(f"\n[FAIL] Request timed out after {TIMEOUT}s.")
            sys.exit(1)
        elapsed = round((time.time() - start) * 1000)

    print(f"\n  Status   : {resp.status_code}  ({elapsed}ms)")

    try:
        body = resp.json()
    except Exception:
        print(f"  Body     : {resp.text[:500]}")
        sys.exit(1)

    if resp.status_code == 200:
        print(f"\n  ✅  SUCCESS")
        v = body.get("verification", {})
        print(f"  Overall Score : {v.get('overall_score')}")
        print(f"  Verdict       : {v.get('verdict')}")
        layers = v.get("layer_results", {})
        for name, res in layers.items():
            status = "✅" if res.get("passed") else "⚠️ "
            err    = f"  ERROR: {res['error']}" if "error" in res else ""
            print(f"    {status} {name}: score={res.get('score')}{err}")
        print(f"\n  Processing time: {body.get('processing_time_ms')}ms")
        print("\n  Full JSON response saved to: test_response.json")
        with open("test_response.json", "w") as f:
            json.dump(body, f, indent=2)
    else:
        print(f"\n  ❌  FAILED")
        detail = body.get("detail", body)
        print(f"  Error detail: {detail}")
        print("\n  Possible causes:")
        if resp.status_code == 400:
            print("    • Brand name not in SUPPORTED_BRANDS list in predict.py")
        elif resp.status_code == 422:
            print("    • File is not a recognised image format, or brand field is missing")
        elif resp.status_code == 500:
            print("    • ML inference crashed — check server terminal for full traceback")
        sys.exit(1)

if __name__ == "__main__":
    run_test()
