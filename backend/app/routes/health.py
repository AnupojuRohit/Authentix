from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def health_check():
    """
    Core health check endpoint.
    Verifies that the API service is up and responding.
    """
    return {"status": "ok", "message": "Authentix backend is running securely natively."}
