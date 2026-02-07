from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random
import time

from .detector import DeepfakeDetector

app = FastAPI()

# Initialize Detector
detector = DeepfakeDetector()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisResponse(BaseModel):
    confidence_score: float
    verdict: str
    explanation: str
    analysis: dict
    artifacts: list
    timeline_data: list
    audio_fingerprint: dict
    spectrogram_base64: str | None = None

class GenerateFakeRequest(BaseModel):
    text: str
    voice_id: str | None = None

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_filename = f"temp_{int(time.time())}_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # Delegate analysis to the Detector class
        result = detector.analyze(temp_filename)
        return result
    finally:
        # Cleanup
        import os
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/generate-fake")
async def generate_fake(request: GenerateFakeRequest):
    return {"message": "Not implemented yet"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
