import os
import torch
import open_clip
import faiss
import numpy as np
from app.config import settings


class ModelLoader:
    def __init__(self):
        self.yolo_model = None
        self.clip_model = None
        self.clip_preprocess = None
        self.faiss_index = None
        self.faiss_labels = None
        self.brand_faiss_indices = {}
        self.max_cached_brand_indices = 5
        self.is_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Model Loader] Device: {self.device}")

    def load_all_models(self):
        if self.is_loaded:
            return

        print("[Model Loader] Initializing AI Models...")
        print(f"[Model Loader] BACKEND_DIR resolved to: {settings.SAVED_MODELS_DIR}")

        # --- Load OpenCLIP ---
        try:
            print("[Model Loader] Loading OpenCLIP ViT-B-32...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='openai'
            )
            self.clip_model = model.eval().to(self.device)
            self.clip_preprocess = preprocess
            print("[Model Loader] OpenCLIP loaded successfully.")
        except Exception as e:
            print(f"[Model Loader] ERROR: OpenCLIP failed to load: {e}")

        # --- Load YOLO ---
        yolo_path = settings.YOLO_MODEL_PATH
        print(f"[Model Loader] Looking for YOLO at: {yolo_path}")

        if os.path.exists(yolo_path):
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(yolo_path)
                print(f"[Model Loader] YOLO loaded from {yolo_path}")
            except Exception as e:
                print(f"[Model Loader] ERROR: YOLO failed to load: {e}")
        else:
            # Try fallback to base yolov8n.pt in ml folder
            fallback = os.path.join(
                os.path.dirname(settings.SAVED_MODELS_DIR), "ml", "yolov8n.pt"
            )
            if os.path.exists(fallback):
                try:
                    from ultralytics import YOLO
                    self.yolo_model = YOLO(fallback)
                    print(f"[Model Loader] YOLO loaded from fallback: {fallback}")
                except Exception as e:
                    print(f"[Model Loader] ERROR: YOLO fallback failed: {e}")
            else:
                print(f"[Model Loader] WARNING: No YOLO model found. Layer 2 will use fallback score.")

        # --- Skip 9GB global FAISS, use per-brand lazily ---
        faiss_dir = settings.FAISS_INDICES_DIR
        if os.path.exists(faiss_dir):
            available = [
                f.replace(".index", "")
                for f in os.listdir(faiss_dir)
                if f.endswith(".index")
            ]
            print(f"[Model Loader] Found {len(available)} brand FAISS indices in {faiss_dir}")
        else:
            print(f"[Model Loader] WARNING: FAISS indices directory not found: {faiss_dir}")

        self.is_loaded = True
        print("[Model Loader] All models initialized.")

    def get_brand_index(self, brand: str):
        """Lazily load and cache brand-specific FAISS index."""
        if brand in self.brand_faiss_indices:
            return self.brand_faiss_indices[brand]

        faiss_dir = settings.FAISS_INDICES_DIR
        if not os.path.exists(faiss_dir):
            return None

        index_path = os.path.join(faiss_dir, f"{brand}.index")
        if not os.path.exists(index_path):
            print(f"[Model Loader] No FAISS index found for brand: {brand} at {index_path}")
            return None

        try:
            idx = faiss.read_index(index_path)
            print(f"[Model Loader] Loaded FAISS index for {brand} ({idx.ntotal} vectors)")
        except Exception as e:
            print(f"[Model Loader] ERROR loading FAISS index for {brand}: {e}")
            return None

        # Bounded cache eviction
        if len(self.brand_faiss_indices) >= self.max_cached_brand_indices:
            oldest = next(iter(self.brand_faiss_indices))
            del self.brand_faiss_indices[oldest]
            print(f"[Model Loader] Evicted {oldest} from FAISS cache")

        self.brand_faiss_indices[brand] = idx
        return idx


model_loader = ModelLoader()