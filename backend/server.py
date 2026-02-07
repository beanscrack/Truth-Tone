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
    import os
    import requests
    from fastapi.responses import FileResponse
    
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return {"error": "Missing ELEVENLABS_API_KEY in backend/.env"}

    # Default Voice ID (Rachel) - you can change this or pass it from frontend
    voice_id = request.voice_id or "21m00Tcm4TlvDq8ikWAM" 
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    
    data = {
        "text": request.text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        # Save audio temporarily
        filename = f"generated_{int(time.time())}.mp3"
        filepath = os.path.join(os.getcwd(), filename)
        
        with open(filepath, "wb") as f:
            f.write(response.content)
            
        # Optional: Analyze the generated file immediately to show results
        analysis_result = detector.analyze(filepath)
        
        # Return both file url (needs static serving setup) and analysis
        # For simplicity in this demo, we'll return the analysis and a message
        # In a real app, serve the file via StaticFiles or upload to S3
        
        # We need to clean up the file eventually, but let's keep it for now so frontend can play it if we serve it
        # For now, let's just return the analysis result of the fake audio
        return {
            "message": "Audio generated successfully",
            "filename": filename,
            "analysis": analysis_result
        }

    except Exception as e:
        print(f"ElevenLabs Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
