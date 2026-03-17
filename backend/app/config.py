import os

class Settings:
    PROJECT_NAME: str = "Authentix AI"
    VERSION: str = "1.0.0"
    
    # Model Paths
    YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "saved_models/yolo_logo_detector.pt")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "saved_models/faiss_index.bin")
    CLIP_EMBEDDINGS_PATH: str = os.getenv("CLIP_EMBEDDINGS_PATH", "saved_models/clip_embeddings.pkl")
    
    # Inference config
    SIMILARITY_THRESHOLD: float = 0.85

settings = Settings()
