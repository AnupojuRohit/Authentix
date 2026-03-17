from pydantic import BaseModel

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    similarity_score: float
