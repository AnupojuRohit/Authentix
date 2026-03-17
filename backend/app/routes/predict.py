from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from app.schemas import PredictionResponse
from app.services.inference_service import inference_runner
from PIL import Image
import io

router = APIRouter()

@router.post("/", response_model=PredictionResponse)
async def predict(
    brand: str = Form(..., description="The brand name to check against"),
    image: UploadFile = File(..., description="The product image to analyze")
):
    """
    Core AI Inference Endpoint.
    1. Validates the incoming image.
    2. Passes to the `inference_service` orchestrator.
    3. Returns the Authentic/Fake prediction, confidence, and similarity score.
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File uploaded is not an image.")

    try:
        # Load the uploaded bytes natively into a PIL Image for processing
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Execute the AI Pipeline
        result = inference_runner.process_image(pil_image, brand)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        print(f"[Predict Route Error] {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
