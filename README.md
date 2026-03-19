# Authentix AI — Fake Brand Detection System

Authentix is an AI-powered platform that detects whether a product is **authentic or counterfeit** using a multi-layer computer vision pipeline combining CLIP embeddings, FAISS similarity search, YOLO logo detection, and Error Level Analysis.

---

## Core Idea

Upload a product image + select a brand → Three-layer AI pipeline analyzes it → Get an **authenticity verdict, confidence score, heatmap, and buying recommendations**

---

## Tech Stack

### Frontend
- Next.js 14
- TypeScript
- Tailwind CSS

### Backend
- FastAPI (Python 3.11)
- Uvicorn ASGI server
- Pydantic v2

### AI / ML
- **OpenCLIP (ViT-B-32)** — Visual embedding extraction
- **FAISS (IndexFlatIP)** — Per-brand cosine similarity search
- **YOLOv8n** — Logo detection and localization
- **ELA (Error Level Analysis)** — Image manipulation detection
- **Brand DNA (Gabor + LBP + HOG)** — Texture fingerprinting

---

## How the Pipeline Works

Every prediction runs three layers in parallel, fused into a single verdict:

```
Input Image
     │
     ├── Layer 1 (60%) ── OpenCLIP embedding → FAISS brand index → cosine similarity score
     │
     ├── Layer 2 (25%) ── YOLOv8 logo detection → confidence score
     │
     └── Layer 3 (15%) ── Error Level Analysis → image integrity score
                                    │
                             Weighted Fusion
                                    │
                          Sigmoid Calibration
                                    │
                    verdict: authentic / fake / suspicious
```

---

## Supported Brands (38 total)

Nike, Adidas, Gucci, Prada, Versace, Valentino, Louis Vuitton, Balenciaga,
Bottega Veneta, Celine, Fendi, Chanel, Hermes, Miu Miu, Burberry, Goyard,
Maison Margiela, Off-White, Rick Owens, Saint Laurent, Givenchy, Moncler,
Converse, Vans, New Balance, Salomon, Asics, Puma, Reebok, Yeezy,
Jordan, Fila, Kith, Lacoste, Timberland, The North Face, Tommy Hilfiger,
Under Armour

---

## Project Structure

```
Authentix/
│
├── backend/
│   ├── app/
│   │   ├── models/
│   │   │   └── model_loader.py       # Singleton: loads CLIP, YOLO, FAISS
│   │   ├── routes/
│   │   │   ├── health.py             # GET /health
│   │   │   └── predict.py            # POST /predict/
│   │   ├── services/
│   │   │   ├── authenticity_service.py   # 3-layer inference pipeline
│   │   │   └── recommendation_service.py # Counterfeit → buy authentic
│   │   ├── config.py                 # Absolute path settings
│   │   └── main.py                   # FastAPI app entry point
│   │
│   ├── ml/
│   │   ├── extract_embeddings.py     # CLIP embedding extraction (run once)
│   │   ├── build_faiss_index.py      # Build per-brand FAISS indices (run once)
│   │   ├── build_brand_dna.py        # Gabor/LBP/HOG brand profiles (run once)
│   │   ├── train_logo_detector.py    # Fine-tune YOLOv8 on logo dataset (run once)
│   │   └── build_embeddings.py       # Legacy embedding builder
│   │
│   ├── saved_models/                 # Model artifacts (not in Git)
│   │   ├── brand_thresholds.json     # Per-brand calibrated similarity thresholds
│   │   ├── brand_dna.pkl             # Brand texture fingerprints
│   │   └── yolo_logo_detector.pt     # Fine-tuned YOLO (generated after training)
│   │
│   ├── faiss_indices/                # Per-brand FAISS indices (not in Git)
│   │   └── {Brand}.index             # e.g. Nike.index, Adidas.index ...
│   │
│   ├── embeddings/                   # Raw brand embeddings (not in Git)
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx
│   │   │   ├── layout.tsx
│   │   │   └── globals.css
│   │   ├── components/
│   │   │   ├── UploadBox.tsx
│   │   │   └── ResultPanel.tsx
│   │   └── services/
│   │       └── api.ts
│   └── package.json
│
├── data/                             # Product catalog JSON (not in Git)
├── dataset/                          # Training images (not in Git)
├── .gitignore
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/AnupojuRohit/Authentix.git
cd Authentix
```

---

### 2. Backend Setup

**Requirements:** Python 3.11, NVIDIA GPU recommended (CPU works but slower)

```bash
# Create a single clean virtual environment
python -m venv backend\.venv

# Activate it (Windows PowerShell)
backend\.venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r backend/requirements.txt
```

**Run the backend:**

```bash
cd backend
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Backend runs at: `http://127.0.0.1:8000`

On successful startup you should see:
```
[Model Loader] Device: cuda
[Model Loader] OpenCLIP loaded successfully.
[Model Loader] YOLO loaded successfully.
[Model Loader] Found 38 brand FAISS indices
[AuthenticityService] Loaded thresholds for 38 brands
INFO: Application startup complete.
```

---

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: `http://localhost:3000`

---

## API Reference

| Endpoint     | Method | Description                              |
|--------------|--------|------------------------------------------|
| `/health`    | GET    | Check if backend is running              |
| `/predict/`  | POST   | Upload image + brand → get verdict       |

### POST `/predict/` — Request

| Field   | Type   | Description                        |
|---------|--------|------------------------------------|
| `brand` | string | Brand name e.g. `Nike`             |
| `image` | file   | Product image (jpg/png/webp)       |

### POST `/predict/` — Response

```json
{
  "verdict": "authentic",
  "confidence": 84.2,
  "authentic_probability": 0.842,
  "fake_probability": 0.158,
  "confidence_level": "high",
  "heatmap_base64": "data:image/png;base64,...",
  "analysis_regions": ["Logo Region", "Stitching Pattern", "Material Texture"],
  "processing_time_ms": 312,
  "layer_scores": {
    "visual_similarity": 88.4,
    "logo_detection": 76.2,
    "image_integrity": 91.0
  },
  "recommendation": null
}
```

---

## ML Pipeline — One-Time Setup

If you have the dataset and want to rebuild all model artifacts from scratch:

```bash
cd backend

# Step 1 — Extract CLIP embeddings for all brands
python ml/extract_embeddings.py

# Step 2 — Build per-brand FAISS indices + calibrate thresholds
python ml/build_faiss_index.py

# Step 3 — Build brand texture DNA profiles
python ml/build_brand_dna.py

# Step 4 — Fine-tune YOLOv8 logo detector (requires dataset, ~45 min on RTX 4050)
python ml/train_logo_detector.py
```

---

## Current Status

| Component              | Status      |
|------------------------|-------------|
| Frontend UI            | ✅ Complete  |
| FastAPI Backend        | ✅ Complete  |
| OpenCLIP + FAISS       | ✅ Working   |
| Per-brand thresholds   | ✅ Calibrated (38 brands) |
| Brand DNA profiles     | ✅ Built     |
| YOLO Logo Detection    | ⚠️ Using base YOLOv8n (fine-tuning pending) |
| GPU Acceleration       | ✅ CUDA (RTX 4050) |
| Recommendation Engine  | ✅ Working   |

---

## Future Improvements

- Fine-tune YOLO on full logo dataset for better Layer 2 accuracy
- Add more brand support beyond 38
- Grad-CAM attention heatmaps (replace center-focus placeholder)
- Cloud deployment (AWS / GCP / Railway)
- Mobile app integration
- Batch image processing endpoint
- Confidence calibration improvements with real fake product data

---

## Important Notes

- `dataset/`, `embeddings/`, `faiss_indices/`, and large `.pt` / `.bin` model files are excluded from Git due to size
- `brand_thresholds.json` and `brand_dna.pkl` are included — these are small pre-computed artifacts
- The system requires the FAISS indices to be present in `backend/faiss_indices/` to function — these must be generated locally using the ML pipeline scripts above
- YOLO currently uses `yolov8n.pt` as fallback until `train_logo_detector.py` is run to generate `yolo_logo_detector.pt`

---

## Author

**Rohit Anupoju**  
B.Tech CSE | AI & ML

---

## Contributing

Feel free to fork this repo and open a PR. Issues and suggestions welcome.
