import time
import traceback
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional, List, Dict
from pydantic import BaseModel

from app.services.authenticity_service import authenticity_service, VerificationResult
from app.services.recommendation_service import recommendation_service, RecommendationResult

class FullInferenceResponse(BaseModel):
    """The exact flat structure requested by the user in Step 6."""
    verdict: str
    confidence: float
    authentic_probability: float
    fake_probability: float
    confidence_level: str
    heatmap_base64: str
    analysis_regions: List[str]
    processing_time_ms: int
    recommendation: Optional[RecommendationResult] = None
    layer_scores: Optional[Dict[str, float]] = None

router = APIRouter()

# Current Active MVP Brands
SUPPORTED_BRANDS = [
    "Nike", "Adidas", "Gucci", "Hoka", "Timberland", 
    "Bottega Veneta", "Celine", "Vans", "Versace", "Valentino",
    "Maison Margiela", "Converse", "Rick Owens", "New Balance",
    "Salomon", "Louis Vuitton", "Puma", "Asics", "Yeezy", "Fendi", "Prada"
]

@router.post("/", response_model=FullInferenceResponse)
async def predict(
    brand: str = Form(..., description="The brand name to check against"),
    image: UploadFile = File(..., description="The product image to analyze")
):
    """
    Overhauled API to return the flat structured response requested by the user.
    """
    # Validation
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="File uploaded is not a valid image format.")
        
    normalized_brand = brand.strip().title().replace(" ", "_")
    
    is_supported = any(supported.lower() == brand.lower() for supported in SUPPORTED_BRANDS)
    if not is_supported:
        raise HTTPException(
            status_code=400, 
            detail=f"Brand '{brand}' is not yet supported. Supported models: {', '.join(SUPPORTED_BRANDS)}"
        )

    try:
        image_bytes = await image.read()
        
        # 1. Verification
        res = authenticity_service.verify(image_bytes, normalized_brand)
        
        # 2. Recommendation
        rec_res = None
        if res.verdict in ["fake", "suspicious"]:
            rec_res = recommendation_service.get_recommendations(brand.strip().title(), res.verdict, image_bytes)
            
        return FullInferenceResponse(
            verdict=res.verdict,
            confidence=res.confidence,
            authentic_probability=res.authentic_probability,
            fake_probability=res.fake_probability,
            confidence_level=res.confidence_level,
            heatmap_base64=res.heatmap_base64,
            analysis_regions=res.analysis_regions,
            processing_time_ms=res.processing_time_ms,
            recommendation=rec_res,
            layer_scores=res.layer_scores
        )
        
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        error_detail = f"{type(e).__name__}: {str(e) or 'Internal error'}"
        print(f"[Predict Error] {error_detail}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {error_detail}")

