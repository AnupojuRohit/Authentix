import os
import torch
import open_clip
import faiss
import numpy as np
from app.config import settings

class ModelLoader:
    """
    Centralized singleton responsible for loading and holding all AI models in memory.
    Prevents reloading models on every API request.
    """
    def __init__(self):
        self.yolo_model = None
        self.clip_model = None
        self.clip_preprocess = None
        self.faiss_index = None
        self.faiss_labels = None
        self.brand_faiss_indices = {}  # Cache loaded brand-specific FAISS indices
        self.brand_faiss_dir = None
        self.max_cached_brand_indices = 4
        self.is_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_all_models(self):
        """
        Loads YOLO, OpenCLIP, and FAISS into memory.
        If paths don't exist yet (pre-dataset), fails gracefully.
        """
        if self.is_loaded:
            return

        print("[Model Loader] Initializing AI Models...")
        
        # Load OpenCLIP (Always available)
        try:
            print("[Model Loader] Loading OpenCLIP (ViT-B-32)...")
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
            self.clip_model = model.eval().to(self.device)
            self.clip_preprocess = preprocess
        except Exception as e:
            print(f"[Model Loader] Failed to load OpenCLIP: {e}")

        # Load YOLO
        if os.path.exists(settings.YOLO_MODEL_PATH):
            print(f"[Model Loader] Loading YOLO from {settings.YOLO_MODEL_PATH}")
            # self.yolo_model = YOLO(settings.YOLO_MODEL_PATH)
        else:
            print(f"[Model Loader] WARNING: YOLO model not found at {settings.YOLO_MODEL_PATH}")

        # Load FAISS (Skipping 9GB Global Index to prevent OOM/Crash on startup)
        print("[Model Loader] Skipping 9GB Global FAISS index for startup stability.")
        print("[Model Loader] Using brand-specific indices lazily instead.")
        self.faiss_index = None
        self.faiss_labels = None
        
        # Detect brand-specific FAISS directory (brand indices are loaded lazily)
        backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidate_dirs = [
            os.path.join(backend_dir, "faiss_indices"),
            os.path.join(backend_dir, "embeddings", "faiss_indices"),
        ]
        self.brand_faiss_dir = next((d for d in candidate_dirs if os.path.exists(d)), None)
        if self.brand_faiss_dir:
            print(f"[Model Loader] Brand FAISS directory detected: {self.brand_faiss_dir}")
            print("[Model Loader] Brand indices will be loaded lazily on demand.")
        else:
            print(
                "[Model Loader] WARNING: Brand FAISS indices directory not found. "
                f"Checked: {candidate_dirs}"
            )
            
        self.is_loaded = True

    def get_brand_index(self, brand: str):
        """Lazily load and cache a brand-specific FAISS index with bounded memory usage."""
        if brand in self.brand_faiss_indices:
            return self.brand_faiss_indices[brand]

        if not self.brand_faiss_dir:
            return None

        index_path = os.path.join(self.brand_faiss_dir, f"{brand}.index")
        if not os.path.exists(index_path):
            return None

        try:
            idx = faiss.read_index(index_path)
        except Exception as e:
            print(f"[Model Loader] WARNING: Failed to load brand index {brand}: {e}")
            return None

        # Simple bounded cache eviction to avoid unbounded RAM growth
        if len(self.brand_faiss_indices) >= self.max_cached_brand_indices:
            oldest_brand = next(iter(self.brand_faiss_indices))
            self.brand_faiss_indices.pop(oldest_brand, None)

        self.brand_faiss_indices[brand] = idx
        return idx

# Global instance
model_loader = ModelLoader()
