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

from fastapi.staticfiles import StaticFiles
import os
# Ensure directory exists for static mount
os.makedirs("backend/generated", exist_ok=True)
app.mount("/files", StaticFiles(directory="backend/generated"), name="files")

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

@app.get("/")
def read_root():
    return {"message": "TruthTone++ API is running", "docs_url": "/docs"}

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    temp_filename = f"temp_{int(time.time())}_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # Delegate analysis to the Detector class (run in threadpool to avoid blocking event loop)
        from fastapi.concurrency import run_in_threadpool
        result = await run_in_threadpool(detector.analyze, temp_filename)
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
    
    # Ensure generated directory exists
    GENERATED_DIR = os.path.join(os.path.dirname(__file__), "generated")
    os.makedirs(GENERATED_DIR, exist_ok=True)

    # Define fallback function
    async def use_fallback_audio():
        print("Using fallback audio (test-audio-fake.mp3)...")
        # Go up one level from backend/ to root, then frontend/public/...
        # backend/server.py is in backend/
        # so os.pardir from server.py dir
        root_dir = os.path.dirname(os.path.dirname(__file__))
        fallback_path = os.path.join(root_dir, "frontend", "public", "test-audio-fake.mp3")
        
        if not os.path.exists(fallback_path):
             return {"error": "ElevenLabs API failed and fallback file not found."}
        
        # Create a new fake file from the fallback
        filename = f"generated_fake_{int(time.time())}.mp3"
        filepath = os.path.join(GENERATED_DIR, filename)
        
        import shutil
        shutil.copy2(fallback_path, filepath)
        
        # Analyze the fallback file
        from fastapi.concurrency import run_in_threadpool
        analysis_result = await run_in_threadpool(detector.analyze, filepath)
        
        return {
            "message": "Audio generation simulated (using demo file due to missing/invalid API key)",
            "filename": filename,
            "analysis": analysis_result
        }

    api_key = os.getenv("ELEVENLABS_API_KEY")
    
    # If no key, go straight to fallback
    if not api_key:
        print("No ElevenLabs API key found. Attempting fallback.")
        return await use_fallback_audio()

    # Default Voice ID (Rachel)
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
        filename = f"generated_fake_{int(time.time())}.mp3"
        filepath = os.path.join(os.getcwd(), "backend", "generated", filename)
        # Ensure dir exists safely (redundant but safe)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, "wb") as f:
            f.write(response.content)
            
        # Optional: Analyze the generated file immediately to show results
        from fastapi.concurrency import run_in_threadpool
        analysis_result = await run_in_threadpool(detector.analyze, filepath)
        
        return {
            "message": "Audio generated successfully",
            "filename": filename,
            "analysis": analysis_result
        }

    except Exception as e:
        print(f"ElevenLabs Error: {e}")
        # On any error (401, quota exceeded, etc), try fallback
        return await use_fallback_audio()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
