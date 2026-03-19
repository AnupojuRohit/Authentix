import time
import traceback
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

from app.services.authenticity_service import authenticity_service, VerificationResult
from app.services.recommendation_service import recommendation_service, RecommendationResult


class FullInferenceResponse(BaseModel):
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

SUPPORTED_BRANDS = [
    "Nike", "Adidas", "Gucci", "Timberland",
    "Bottega_Veneta", "Celine", "Vans", "Versace", "Valentino",
    "Maison_Margiela", "Converse", "Rick_Owens", "New_Balance",
    "Salomon", "Louis_Vuitton", "Puma", "Asics", "Yeezy",
    "Fendi", "Prada", "Balenciaga", "Burberry", "Chanel",
    "Fila", "Goyard", "Gucci", "Hermes", "Jordan", "Kith",
    "Lacoste", "Miu_Miu", "Moncler", "Off-White", "Reebok",
    "Saint_Laurent", "The_North_Face", "Tommy_Hilfiger",
    "Under_Armour"
]


@router.post("/", response_model=FullInferenceResponse)
async def predict(
    brand: str = Form(..., description="The brand name to check against"),
    image: UploadFile = File(..., description="The product image to analyze")
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail="File uploaded is not a valid image format."
        )

    # Normalize brand name to match FAISS index filenames
    normalized_brand = brand.strip().title().replace(" ", "_")

    is_supported = any(
        s.lower() == normalized_brand.lower() for s in SUPPORTED_BRANDS
    )
    if not is_supported:
        raise HTTPException(
            status_code=400,
            detail=f"Brand '{brand}' is not supported. Supported: {', '.join(SUPPORTED_BRANDS)}"
        )

    try:
        image_bytes = await image.read()

        res = authenticity_service.verify(image_bytes, normalized_brand)

        rec_res = None
        if res.verdict in ["fake", "suspicious"]:
            rec_res = recommendation_service.get_recommendations(
                brand.strip().title(), res.verdict, image_bytes
            )

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
        print(f"[Predict Error] {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(e).__name__}: {str(e)}"
        )