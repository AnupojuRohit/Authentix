import os

# Absolute path to the backend directory regardless of where you run from
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Settings:
    PROJECT_NAME: str = "Authentix AI"
    VERSION: str = "1.0.0"

    # Absolute model paths - works no matter where you run uvicorn from
    YOLO_MODEL_PATH: str = os.path.join(BACKEND_DIR, "saved_models", "yolo_logo_detector.pt")
    FAISS_INDEX_PATH: str = os.path.join(BACKEND_DIR, "saved_models", "faiss_index.bin")
    FAISS_INDICES_DIR: str = os.path.join(BACKEND_DIR, "faiss_indices")
    SAVED_MODELS_DIR: str = os.path.join(BACKEND_DIR, "saved_models")
    EMBEDDINGS_DIR: str = os.path.join(BACKEND_DIR, "embeddings")

    # Inference config
    SIMILARITY_THRESHOLD: float = 0.85


settings = Settings()