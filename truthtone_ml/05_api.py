"""
05_api.py
FastAPI inference server for TruthTone++.

Endpoints:
  POST /analyze          - Analyze audio file for deepfakes
  POST /analyze-segment  - Analyze with per-segment heatmap data
  POST /generate-fake    - Generate deepfake using ElevenLabs
  GET  /health           - Health check

Usage:
    python 05_api.py --model checkpoints/best_model.pt --port 8000

The /analyze endpoint returns:
{
    "overall_score": 87.3,       // % confidence it's REAL
    "verdict": "LIKELY REAL",
    "segments": [...],            // per-timestamp scores for heatmap
    "spectrogram": [[...]],       // 2D array for 3D visualization
    "frequencies": [...],         // raw frequency data
    "gemini_explanation": "...",  // human-readable explanation
    "audio_hash": "sha256:..."   // for blockchain verification
}
"""
import os
import io
import hashlib
import tempfile
import argparse
from dotenv import load_dotenv
import numpy as np
import torch
import librosa
from torch.amp import autocast
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from config import (
    SAMPLE_RATE, AUDIO_DURATION, N_MELS, N_FFT, HOP_LENGTH,
    SPEC_IMAGE_SIZE, SEGMENT_LENGTH, SEGMENT_HOP,
    GEMINI_MODEL, API_HOST, API_PORT, MAX_UPLOAD_SIZE_MB, CLASS_NAMES
)
from model import build_model

# ── App Setup ──
app = FastAPI(
    title="TruthTone++ API",
    description="Deepfake audio detection with custom ML model",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ──
model = None
device = None
gemini_model = None


class AnalysisResult(BaseModel):
    overall_score: float           # 0-100, % confidence it's REAL
    verdict: str                   # REAL / LIKELY REAL / UNCERTAIN / LIKELY FAKE / FAKE
    segments: list                 # per-segment scores for heatmap
    spectrogram: list              # 2D array for 3D viz (downsampled)
    spectrogram_viz: list          # 20x20 normalized 0-1 array for 3D mesh
    frequencies: list              # raw frequency magnitudes
    gemini_explanation: str        # human-readable explanation
    analysis: dict                 # Gemini-generated attributes (breathing, prosody, etc.)
    artifacts: list                # Gemini-detected artifacts with timestamps
    audio_hash: str                # SHA-256 hash
    duration: float                # audio duration in seconds
    sample_rate: int


def audio_to_spectrogram(audio_bytes):
    """Convert audio bytes to mel spectrogram."""
    # Save to temp file (librosa needs a file path or file-like object)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name
    
    try:
        y, sr = librosa.load(temp_path, sr=SAMPLE_RATE, mono=True)
    finally:
        os.unlink(temp_path)
    
    duration = len(y) / sr
    
    # Full spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    return y, sr, mel_db, duration


def spectrogram_to_input(mel_db):
    """Convert spectrogram to model input tensor."""
    from PIL import Image
    
    # Normalize to 0-255
    mel_norm = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8) * 255)
    mel_uint8 = mel_norm.astype(np.uint8)
    
    # Resize to model input size
    img = Image.fromarray(mel_uint8).resize((SPEC_IMAGE_SIZE, SPEC_IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch and channel dims: (1, 1, 224, 224)
    tensor = torch.FloatTensor(arr).unsqueeze(0).unsqueeze(0)
    return tensor


def classify_audio(y, sr):
    """Run full audio through model."""
    # Pad/trim
    target_len = int(sr * AUDIO_DURATION)
    if len(y) < target_len:
        y_padded = np.pad(y, (0, target_len - len(y)))
    else:
        y_padded = y[:target_len]
    
    mel = librosa.feature.melspectrogram(
        y=y_padded, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    input_tensor = spectrogram_to_input(mel_db).to(device)
    
    with torch.no_grad():
        with autocast(device.type):
            output = model(input_tensor)
    
    probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    real_prob = float(probs[0]) * 100  # % real
    
    return real_prob


def classify_segments(y, sr):
    """Run sliding window analysis for per-segment scores."""
    segments = []
    seg_samples = int(SEGMENT_LENGTH * sr)
    hop_samples = int(SEGMENT_HOP * sr)
    target_len = int(AUDIO_DURATION * sr)
    
    pos = 0
    while pos + seg_samples <= len(y):
        segment = y[pos:pos + seg_samples]
        
        # Pad segment to AUDIO_DURATION for model compatibility
        seg_padded = np.pad(segment, (0, max(0, target_len - len(segment))))[:target_len]
        
        mel = librosa.feature.melspectrogram(
            y=seg_padded, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        input_tensor = spectrogram_to_input(mel_db).to(device)
        
        with torch.no_grad():
            with autocast(device.type):
                output = model(input_tensor)
        
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        real_score = float(probs[0]) * 100
        
        segments.append({
            "start": round(pos / sr, 2),
            "end": round((pos + seg_samples) / sr, 2),
            "score": round(real_score, 1),
        })
        
        pos += hop_samples
    
    return segments


def get_verdict(score):
    """Convert score to human-readable verdict."""
    if score >= 85:
        return "REAL"
    elif score >= 65:
        return "LIKELY REAL"
    elif score >= 40:
        return "UNCERTAIN"
    elif score >= 20:
        return "LIKELY FAKE"
    else:
        return "FAKE"


async def get_gemini_explanation(score, verdict, segments, duration):
    """Use Gemini to generate human-readable explanation."""
    try:
        import google.generativeai as genai
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return "Gemini API key not configured. Set GEMINI_API_KEY environment variable."
        
        genai.configure(api_key=api_key)
        gmodel = genai.GenerativeModel(GEMINI_MODEL)
        
        # Find flagged segments (low real score = suspicious)
        flagged = [s for s in segments if s["score"] < 60]
        flagged_str = ", ".join([f"{s['start']}-{s['end']}s (score: {s['score']}%)" for s in flagged[:5]])
        
        prompt = f"""You are an audio forensics expert explaining deepfake detection results.

Our ML model (a CNN trained on mel spectrograms) analyzed a {duration:.1f}-second audio clip:
- Overall confidence: {score:.1f}% real
- Verdict: {verdict}
- Suspicious segments: {flagged_str if flagged else 'None detected'}
- Total segments analyzed: {len(segments)}

In 2-3 concise sentences, explain why this audio appears {verdict.lower()}. 
Reference specific artifacts like prosody irregularities, breathing patterns, 
pitch transitions, spectral smoothness, or formant consistency.
Be specific about timestamps if there are flagged segments.
Write for a general audience. Do not use bullet points."""

        response = gmodel.generate_content(prompt)
        return response.text.strip()
    
    except ImportError:
        return "Gemini SDK not installed. Run: pip install google-generativeai"
    except Exception as e:
        return f"Gemini explanation unavailable: {str(e)}"


async def get_gemini_analysis_attributes(score, verdict, segments, duration):
    """Use Gemini to generate structured analysis attributes and artifacts."""
    default_analysis = {
        "breathing": "unknown",
        "prosody_variation": "unknown",
        "frequency_spectrum": "unknown",
        "speaking_rhythm": "unknown"
    }
    default_artifacts = []

    try:
        import json
        import google.generativeai as genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return default_analysis, default_artifacts

        genai.configure(api_key=api_key)
        gmodel = genai.GenerativeModel(GEMINI_MODEL)

        flagged = [s for s in segments if s["score"] < 60]
        flagged_str = ", ".join([f"{s['start']}-{s['end']}s (score: {s['score']}%)" for s in flagged[:5]])

        prompt = f"""You are an audio forensics expert. Our CNN model analyzed a {duration:.1f}-second audio clip:
- Overall confidence: {score:.1f}% real
- Verdict: {verdict}
- Suspicious segments: {flagged_str if flagged else 'None detected'}
- Total segments analyzed: {len(segments)}

Return ONLY a JSON object (no code blocks) with this EXACT structure:
{{
    "analysis": {{
        "breathing": "natural or unnatural",
        "prosody_variation": "high or low",
        "frequency_spectrum": "organic or mechanical",
        "speaking_rhythm": "human_inconsistency or consistent"
    }},
    "artifacts": [
        {{"timestamp": 1.5, "type": "glitch", "description": "brief metallic sound"}}
    ]
}}

For the analysis fields:
- breathing: "natural" if score >= 60, "unnatural" if below
- prosody_variation: "high" if score >= 60, "low" if below
- frequency_spectrum: "organic" if score >= 60, "mechanical" if below
- speaking_rhythm: "human_inconsistency" if score >= 60, "consistent" if below

For artifacts: list any suspicious segments as artifacts with their timestamps. If no suspicious segments, return an empty array."""

        response = gmodel.generate_content(prompt)
        response_text = response.text.replace('```json', '').replace('```', '').strip()

        data = json.loads(response_text)
        analysis = data.get("analysis", default_analysis)
        artifacts = data.get("artifacts", default_artifacts)
        return analysis, artifacts

    except Exception:
        # Fallback: derive attributes from the CNN score
        if score >= 60:
            analysis = {
                "breathing": "natural",
                "prosody_variation": "high",
                "frequency_spectrum": "organic",
                "speaking_rhythm": "human_inconsistency"
            }
        else:
            analysis = {
                "breathing": "unnatural",
                "prosody_variation": "low",
                "frequency_spectrum": "mechanical",
                "speaking_rhythm": "consistent"
            }
        return analysis, default_artifacts


def compute_spectrogram_viz(mel_db):
    """Downsample spectrogram to 20x20 normalized 0-1 grid for 3D visualization."""
    from PIL import Image

    # Normalize to 0-1
    mel_min = mel_db.min()
    mel_max = mel_db.max()
    norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-8)

    # Resize to 20x20 using PIL
    img = Image.fromarray((norm * 255).astype(np.uint8))
    img_resized = img.resize((20, 20), Image.BILINEAR)
    result = np.array(img_resized, dtype=np.float32) / 255.0

    return result.tolist()


def compute_audio_hash(audio_bytes):
    """SHA-256 hash of the raw audio file."""
    return f"sha256:{hashlib.sha256(audio_bytes).hexdigest()}"


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze an audio file for deepfake detection.
    
    Accepts: .wav, .mp3, .flac, .m4a, .ogg
    Returns: Detection results with confidence score, segments, and visualization data.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(400, "No file provided")
    
    ext = Path(file.filename).suffix.lower()
    if ext not in {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"}:
        raise HTTPException(400, f"Unsupported format: {ext}. Use .wav, .mp3, .flac, .m4a, or .ogg")
    
    # Read file
    audio_bytes = await file.read()
    
    if len(audio_bytes) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large. Max: {MAX_UPLOAD_SIZE_MB}MB")
    
    # Hash the original file
    audio_hash = compute_audio_hash(audio_bytes)
    
    # Convert to audio
    y, sr, mel_db, duration = audio_to_spectrogram(audio_bytes)
    
    # Overall classification
    overall_score = classify_audio(y, sr)
    verdict = get_verdict(overall_score)
    
    # Per-segment analysis
    segments = classify_segments(y, sr)
    
    # Gemini explanation and analysis attributes (run concurrently)
    import asyncio
    explanation, (analysis, artifacts) = await asyncio.gather(
        get_gemini_explanation(overall_score, verdict, segments, duration),
        get_gemini_analysis_attributes(overall_score, verdict, segments, duration),
    )

    # Prepare spectrogram data for 3D visualization (downsample for JSON transfer)
    spec_downsampled = mel_db[::2, ::4].tolist()  # reduce size

    # 20x20 normalized spectrogram for the 3D mesh viz
    spec_viz = compute_spectrogram_viz(mel_db)

    # Frequency magnitudes (for visualization)
    freqs = np.abs(np.fft.rfft(y[:sr]))  # first second
    freq_downsampled = freqs[::10].tolist()[:200]  # top 200 points

    return AnalysisResult(
        overall_score=round(overall_score, 1),
        verdict=verdict,
        segments=segments,
        spectrogram=spec_downsampled,
        spectrogram_viz=spec_viz,
        frequencies=freq_downsampled,
        gemini_explanation=explanation,
        analysis=analysis,
        artifacts=artifacts,
        audio_hash=audio_hash,
        duration=round(duration, 2),
        sample_rate=sr,
    )


@app.post("/generate-fake")
async def generate_fake(
    text: str = "Hello, this is a test of deepfake voice generation.",
    voice_file: Optional[UploadFile] = File(None),
):
    """
    Generate a deepfake audio sample using ElevenLabs.
    Optionally clone a voice from an uploaded audio file.
    """
    try:
        from elevenlabs import ElevenLabs
        
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise HTTPException(500, "ELEVENLABS_API_KEY not set")
        
        client = ElevenLabs(api_key=api_key)
        
        if voice_file:
            # Clone voice from uploaded audio
            voice_bytes = await voice_file.read()
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(voice_bytes)
                voice_path = f.name
            
            try:
                # Use instant voice clone
                voice = client.clone(
                    name="hackathon_clone",
                    files=[voice_path],
                    description="Hackathon voice clone"
                )
                voice_id = voice.voice_id
            finally:
                os.unlink(voice_path)
        else:
            # Use a default voice
            voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel
        
        # Generate speech
        audio = client.generate(
            text=text,
            voice=voice_id,
            model="eleven_multilingual_v2"
        )
        
        # Collect audio bytes
        audio_bytes = b"".join(audio)
        
        # Return as audio file
        from fastapi.responses import Response
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=generated_fake.mp3"}
        )
    
    except ImportError:
        raise HTTPException(500, "elevenlabs package not installed. Run: pip install elevenlabs")
    except Exception as e:
        raise HTTPException(500, f"Generation failed: {str(e)}")


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


def load_model_for_api(model_path):
    """Load model for inference."""
    global model, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path} on {device}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded (epoch {checkpoint.get('epoch', '?')}, "
          f"val_acc: {checkpoint.get('val_acc', '?'):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruthTone++ API Server")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--port", type=int, default=API_PORT)
    parser.add_argument("--host", type=str, default=API_HOST)
    args = parser.parse_args()
    
    load_dotenv()
    
    load_model_for_api(args.model)
    
    print(f"\nStarting API server on {args.host}:{args.port}")
    print(f"  Docs: http://localhost:{args.port}/docs")
    
    uvicorn.run(app, host=args.host, port=args.port)
