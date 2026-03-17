from PIL import Image
import torch
import numpy as np
from app.models.model_loader import model_loader

class EmbeddingService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Takes the cropped logo image and generates a normalized semantic 
        vector embedding using the OpenCLIP model.
        """
        if model_loader.clip_model is None or model_loader.clip_preprocess is None:
            # Fallback to dummy data if model failed to load in dev
            return np.random.rand(1, 512).astype('float32')
            
        transformed_image = model_loader.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = model_loader.clip_model.encode_image(transformed_image)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            
        return embedding.cpu().numpy().astype('float32')
