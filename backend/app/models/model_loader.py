import os
import torch
import open_clip
import faiss
import pickle
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

        # Load FAISS
        if os.path.exists(settings.FAISS_INDEX_PATH):
            print(f"[Model Loader] Loading FAISS index from {settings.FAISS_INDEX_PATH}")
            self.faiss_index = faiss.read_index(settings.FAISS_INDEX_PATH)
            
            labels_path = settings.FAISS_INDEX_PATH.replace("faiss_index.bin", "labels.pkl")
            if os.path.exists(labels_path):
                with open(labels_path, "rb") as f:
                    self.faiss_labels = pickle.load(f)
        else:
            print(f"[Model Loader] WARNING: FAISS index not found at {settings.FAISS_INDEX_PATH}")
            
        self.is_loaded = True

# Global instance
model_loader = ModelLoader()
