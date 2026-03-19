import io
import json
import os
import torch
import open_clip
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from PIL import Image

from app.models.model_loader import model_loader

class RecommendationResult(BaseModel):
    """Fallback recommendation data when an item is deemed counterfeit/suspicious."""
    message: str
    official_store_url: str
    product_matches: List[Dict[str, Any]]
    authorized_retailers: List[Dict[str, str]]
    warning_message: str
    price_comparison: Dict[str, Any]

class RecommendationService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.catalog = self._load_catalog()
        # Cache tokenizer at class level to avoid reinstantiation
        try:
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        except Exception as e:
            print(f"[RecommendationService] Failed to load tokenizer: {e}")
            self.tokenizer = None
        
    def _load_catalog(self) -> Dict[str, Any]:
        service_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(service_dir)))
        catalog_path = os.path.join(project_root, "data", "product_catalog.json")
        try:
            with open(catalog_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[RecommendationService] WARNING: Catalog not found at {catalog_path}")
            return {}
        except Exception as e:
            print(f"[RecommendationService] ERROR loading catalog: {e}")
            return {}

    def get_recommendations(self, brand: str, verdict: str, image_bytes: bytes) -> Optional[RecommendationResult]:
        """
        Uses zero-shot CLIP classification to identify the specific type of shoe
        and returns matching official recommendations from the curated database.
        """
        if brand not in self.catalog:
            return None
            
        catalog_entry = self.catalog[brand]
        
        # 1. Zero-Shot Product Classification
        shoe_types = [
            "running shoe", "basketball shoe", "lifestyle sneaker", 
            "boot", "high-top", "low-top"
        ]
        
        predicted_type = "shoe"
        if model_loader.clip_model and self.tokenizer:
            try:
                pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                inputs = model_loader.clip_preprocess(pil_img).unsqueeze(0).to(self.device)
                
                text_prompts = [f"a photo of a {brand} {st}" for st in shoe_types]
                text_tokens = self.tokenizer(text_prompts).to(self.device)
                
                with torch.no_grad():
                    image_features = model_loader.clip_model.encode_image(inputs)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    text_features = model_loader.clip_model.encode_text(text_tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    
                    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    
                top_prob, top_idx = text_probs[0].topk(1)
                predicted_type = shoe_types[top_idx.item()]
            except Exception as e:
                print(f"[RecommendationService] Classification failed, falling back to basic recommendation: {e}")

        # Construct payload
        msg = f"This appears to be a counterfeit {brand} {predicted_type}. Buy authentic here:"
        if verdict == "suspicious":
            msg = f"We detected anomalies in this {brand} {predicted_type}. Consider buying from verified retailers:"

        return RecommendationResult(
            message=msg,
            official_store_url=catalog_entry.get("official_store", ""),
            product_matches=catalog_entry.get("flagship_products", []),
            authorized_retailers=catalog_entry.get("authorized_retailers", []),
            warning_message="Counterfeit goods often use toxic adhesives and lack structural support.",
            price_comparison={
                "estimated_fake_price": "Unknown (High Risk)",
                "authentic_price_range": catalog_entry.get("flagship_products", [{}])[0].get("price_range", "Retail"),
                "why_worth_it": "Authentic shoes offer warranties, ethical manufacturing, and durability."
            }
        )

recommendation_service = RecommendationService()
