from PIL import Image
from app.services.logo_detection_service import LogoDetectionService
from app.services.embedding_service import EmbeddingService
from app.services.similarity_service import SimilarityService
from app.config import settings

class InferenceService:
    def __init__(self):
        print("Initializing Inference Pipeline Architecture...")
        self.logo_detector = LogoDetectionService()
        self.embedder = EmbeddingService()
        self.similarity_engine = SimilarityService()

    def process_image(self, image: Image.Image, brand: str) -> dict:
        """
        The central ML orchestration function.
        1. Detects & crops logo (YOLOv8)
        2. Generates semantic vector (OpenCLIP)
        3. Computes similarity to authentic dataset (FAISS)
        4. Calculates final prediction based on threshold
        """
        # Step 1: Logo Detection
        cropped_logo = self.logo_detector.detect_and_crop(image)
        
        # Step 2: Feature Extraction
        vector_embedding = self.embedder.generate_embedding(cropped_logo)
        
        # Step 3: Similarity Search
        sim_score = self.similarity_engine.compare_embedding(vector_embedding, brand)
        
        # Step 4: Decision Logic
        is_authentic = sim_score >= settings.SIMILARITY_THRESHOLD
        
        # Calculate a human-readable confidence percentage (0-100)
        confidence = round(sim_score * 100, 2)
        
        result = {
            "prediction": "Authentic" if is_authentic else "Fake",
            "confidence": confidence,
            "similarity_score": round(sim_score, 4)
        }
        
        return result

# Singleton to prevent reloading models every request
inference_runner = InferenceService()
