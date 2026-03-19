import io
import asyncio
from fastapi import UploadFile, Form
from app.routes.predict import predict

async def debug_predict():
    print("Debug: Simulating predict request...")
    # Mock UploadFile
    img_data = b"dummy image data"
    upload_file = UploadFile(filename="test.jpg", file=io.BytesIO(img_data), content_type="image/jpeg")
    
    try:
        # We need to mock the 'Form' behavior or just call it directly
        response = await predict(brand="Nike", image=upload_file)
        print("Debug: SUCCESS!")
        print(response)
    except Exception as e:
        print(f"Debug: FAILED with {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_predict())
