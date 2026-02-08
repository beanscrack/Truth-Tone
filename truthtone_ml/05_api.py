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

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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


def _prepare_input(y_clip, sr):
    """Convert audio clip to model input tensor (no grad, no autocast)."""
    target_len = int(sr * AUDIO_DURATION)
    if len(y_clip) < target_len:
        y_clip = np.pad(y_clip, (0, target_len - len(y_clip)))
    else:
        y_clip = y_clip[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y_clip, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return spectrogram_to_input(mel_db)


def classify_audio(y, sr):
    """Run full audio through model."""
    input_tensor = _prepare_input(y, sr).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    probs = torch.softmax(output, dim=1).float().cpu().numpy()[0]
    real_prob = float(probs[0]) * 100  # % real

    return real_prob


def classify_full_audio_scan(y, sr):
    """
    Scan the entire audio file with 4s windows (matching training duration)
    and aggregate predictions for a global score.
    """
    window_len = int(sr * AUDIO_DURATION) # 4s
    hop_len = int(window_len / 2)         # 2s overlap
    
    # If shorter than 4s, just use the single prediction
    if len(y) < window_len:
        return classify_audio(y, sr)
        
    # Create overlapping windows
    tensors = []
    
    # If exactly 4s, range might be empty with default logic, handle carefully
    # We want at least one window
    if len(y) == window_len:
        positions = [0]
    else:
        positions = range(0, len(y) - window_len + 1, hop_len)
    
    for p in positions:
        segment = y[p : p + window_len]
        tensors.append(_prepare_input(segment, sr))
        
    if not tensors: # Should not happen given check above, but safety first
        return classify_audio(y, sr)

    batch = torch.cat(tensors, dim=0).to(device)
    
    with torch.no_grad():
        output = model(batch)
        
    probs = torch.softmax(output, dim=1).float().cpu().numpy()
    real_probs = probs[:, 0] * 100
    
    # Aggregation strategy:
    # The model detects "fake" artifacts. If ANY segment is strongly fake, 
    # the file is likely fake. Averaging would dilute a brief glitch.
    
    min_score = np.min(real_probs) # The most "fake" segment score
    avg_score = np.mean(real_probs)
    
    # If we find a very suspicious segment (< 50% real), use that minimum score
    # to warn the user. Otherwise, average the scores for stability.
    if min_score < 50:
        return float(min_score)
    else:
        return float(avg_score)


def classify_segments(y, sr):
    """Run batched sliding window analysis for per-segment scores."""
    seg_samples = int(SEGMENT_LENGTH * sr)
    hop_samples = int(SEGMENT_HOP * sr)

    # Collect all segment positions
    positions = []
    pos = 0
    while pos + seg_samples <= len(y):
        positions.append(pos)
        pos += hop_samples

    if not positions:
        return []

    # Prepare all inputs and batch them
    tensors = []
    for p in positions:
        segment = y[p:p + seg_samples]
        tensors.append(_prepare_input(segment, sr))

    batch = torch.cat(tensors, dim=0).to(device)

    # Single batched inference
    with torch.no_grad():
        output = model(batch)

    all_probs = torch.softmax(output, dim=1).float().cpu().numpy()

    segments = []
    for i, p in enumerate(positions):
        segments.append({
            "start": round(p / sr, 2),
            "end": round((p + seg_samples) / sr, 2),
            "score": round(float(all_probs[i][0]) * 100, 1),
        })
    
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
        "breathing": "Natural or Unnatural",
        "prosody_variation": "High or Low",
        "frequency_spectrum": "Organic or Mechanical",
        "speaking_rhythm": "Human Inconsistency or Consistent"
    }},
    "artifacts": [
        {{"timestamp": 1.5, "type": "glitch", "description": "brief metallic sound"}}
    ]
}}

For the analysis fields:
- breathing: "Natural" if score >= 60, "Unnatural" if below
- prosody_variation: "High" if score >= 60, "Low" if below
- frequency_spectrum: "Organic" if score >= 60, "Mechanical" if below
- speaking_rhythm: "Human Inconsistency" if score >= 60, "Consistent" if below

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
                "breathing": "Natural",
                "prosody_variation": "High",
                "frequency_spectrum": "Organic",
                "speaking_rhythm": "Human Inconsistency"
            }
        else:
            analysis = {
                "breathing": "Unnatural",
                "prosody_variation": "Low",
                "frequency_spectrum": "Mechanical",
                "speaking_rhythm": "Consistent"
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

    try:
        # Hash the original file
        audio_hash = compute_audio_hash(audio_bytes)

        # Convert to audio
        y, sr, mel_db, duration = audio_to_spectrogram(audio_bytes)

        # Overall classification (scan whole file)
        overall_score = classify_full_audio_scan(y, sr)
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
            sample_rate=int(sr),
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Analysis error: {str(e)}")


@app.post("/generate-fake")
async def generate_fake(
    text: str = Form("Hello, this is a test of deepfake voice generation."),
    voice_file: Optional[UploadFile] = File(None),
):
    """
    Generate a deepfake audio sample using ElevenLabs, then analyze it immediately.
    Returns JSON with analysis results and base64 audio.
    """
    import base64
    
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
        audio_generator = client.generate(
            text=text,
            voice=voice_id,
            model="eleven_multilingual_v2"
        )
        
        # Collect audio bytes
        audio_bytes = b"".join(audio_generator)
        
        # 1. Analyze the generated audio
        # We need to compute spectrograms etc. manually here since we have bytes, not a file upload object suitable for the other endpoint
        # But we can reuse the helper functions
        
        # Hash
        audio_hash = compute_audio_hash(audio_bytes)
        
        # Convert to audio/spectrogram
        y, sr, mel_db, duration = audio_to_spectrogram(audio_bytes)
        
        # Classification
        overall_score = classify_full_audio_scan(y, sr)
        verdict = get_verdict(overall_score)
        
        # Segments
        segments = classify_segments(y, sr)
        
        # Gemini (run concurrently)
        import asyncio
        explanation, (analysis, artifacts) = await asyncio.gather(
            get_gemini_explanation(overall_score, verdict, segments, duration),
            get_gemini_analysis_attributes(overall_score, verdict, segments, duration),
        )

        # Viz data
        spec_downsampled = mel_db[::2, ::4].tolist()
        spec_viz = compute_spectrogram_viz(mel_db)
        freqs = np.abs(np.fft.rfft(y[:sr]))
        freq_downsampled = freqs[::10].tolist()[:200]

        analysis_result = {
            "overall_score": round(overall_score, 1),
            "verdict": verdict,
            "segments": segments,
            "spectrogram": spec_downsampled,
            "spectrogram_viz": spec_viz,
            "frequencies": freq_downsampled,
            "gemini_explanation": explanation,
            "analysis": analysis,
            "artifacts": artifacts,
            "audio_hash": audio_hash,
            "duration": round(duration, 2),
            "sample_rate": sr,
        }
        
        # 2. Return JSON with analysis AND audio data (base64)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return {
            "status": "success",
            "analysis": analysis_result,
            "audio_base64": audio_base64, # Frontend can play this: <audio src={`data:audio/mpeg;base64,${audio_base64}`} />
            "message": "Generated and analyzed successfully"
        }

    except ImportError:
        raise HTTPException(500, "elevenlabs package not installed. Run: pip install elevenlabs")
    except Exception as e:
        print(f"Generate error: {e}")
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


HF_REPO_ID = "beanscrack/truthtone-model"
HF_FILENAME = "best_model.pt"


def download_model_from_hf():
    """Download model checkpoint from Hugging Face if not cached locally."""
    from huggingface_hub import hf_hub_download

    print(f"Downloading model from HuggingFace: {HF_REPO_ID}/{HF_FILENAME}")
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    print(f"Model cached at: {model_path}")
    return model_path


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
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint (auto-downloads from HuggingFace if not provided)")
    parser.add_argument("--port", type=int, default=API_PORT)
    parser.add_argument("--host", type=str, default=API_HOST)
    args = parser.parse_args()

    load_dotenv()

    model_path = args.model if args.model else download_model_from_hf()
    load_model_for_api(model_path)

    print(f"\nStarting API server on {args.host}:{args.port}")
    print(f"  Docs: http://localhost:{args.port}/docs")

    uvicorn.run(app, host=args.host, port=args.port)
