import os
import random
import time
import numpy as np
import librosa
import google.generativeai as genai
from dotenv import load_dotenv
import json

load_dotenv()

class DeepfakeDetector:
    def __init__(self):
        """
        Initialize the Gemini model and other configurations.
        """
        print("Initializing DeepfakeDetector with Gemini...")
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("WARNING: GEMINI_API_KEY not found in environment variables. Analysis will fail.")
        else:
            genai.configure(api_key=api_key)
            # Use Gemini 1.5 Flash for speed and multimodal capabilities
            self.model = genai.GenerativeModel('gemini-1.5-flash')

    def analyze(self, audio_path: str):
        """
        Analyze the audio file using Librosa for signal processing and Gemini for semantic analysis.
        """
        try:
            # 1. Librosa Analysis (Signal Processing)
            y, sr = librosa.load(audio_path)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Extract features for visualization and heuristics
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            # Simplified Audio Fingerprint for Visualization
            # Generate a 2D spectrogram (Frequency x Time) for 3D mesh
            # We downsample to a small grid (e.g., 20x20) to keep JSON payload light
            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
            
            # Normalize to 0-1 range
            norm_spectrogram = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min())
            
            # Resize/Resample to 20 time steps for a consistent 20x20 grid
            # Using simple interpolation
            target_time_steps = 20
            current_time_steps = norm_spectrogram.shape[1]
            indices = np.linspace(0, current_time_steps - 1, target_time_steps).astype(int)
            small_spectrogram = norm_spectrogram[:, indices].tolist() # 2D List (20x20)

            # 2. Gemini Analysis (Upload and Prompt)
            # Upload the file to Gemini
            myfile = genai.upload_file(audio_path)
            
            prompt = """
            Analyze this audio file for signs of being AI-generated (deepfake) or real human speech.
            Focus on:
            1. Breathing patterns (natural pauses vs unnatural silence).
            2. Prosody and intonation (emotional variation vs robotic flatness).
            3. Anomalies (glitches, metallic sounds, phase issues).
            4. Consistency in background noise.
            
            Return ONLY a JSON object with this EXACT structure (do not use code blocks):
            {
                "confidence_score": 0.85, 
                "verdict": "REAL", 
                "explanation": "concise explanation observing breathing and prosody...",
                "analysis": {
                    "breathing": "natural/unnatural",
                    "prosody_variation": "high/low",
                    "frequency_spectrum": "organic/mechanical",
                    "speaking_rhythm": "consistent/irregular"
                },
                "artifacts": [
                    {"timestamp": 1.5, "type": "glitch", "description": "brief metallic sound"}
                ]
            }
            Valid verdicts: REAL, LIKELY REAL, UNCERTAIN, LIKELY FAKE, FAKE.
            Confidence should be 0.0 to 1.0 (higher means more likely REAL).
            If it sounds fake, confidence should be low (<0.5).
            """
            
            result = self.model.generate_content([myfile, prompt])
            response_text = result.text.replace('```json', '').replace('```', '').strip()
            
            try:
                gemini_data = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback if Gemini returns malformed JSON
                print("Gemini returned invalid JSON:", response_text)
                gemini_data = {
                    "confidence_score": 0.5,
                    "verdict": "UNCERTAIN",
                    "explanation": "AI analysis returned an unstructured response.",
                    "analysis": {
                        "breathing": "unknown",
                        "prosody_variation": "unknown",
                        "frequency_spectrum": "unknown",
                        "speaking_rhythm": "unknown"
                    },
                    "artifacts": []
                }

            # 3. Combine Data
            # Generate fake timeline data for visualization since Gemini doesn't give sec-by-sec confidence yet
            timeline_data = []
            steps = 20 # fixed number of steps for graph
            base_conf = gemini_data.get("confidence_score", 0.5)
            for i in range(steps):
                timeline_data.append({
                    "time": (duration / steps) * i,
                    "confidence": max(0, min(1, base_conf + random.uniform(-0.1, 0.1)))
                })

            return {
                "confidence_score": gemini_data.get("confidence_score", 0.5),
                "verdict": gemini_data.get("verdict", "UNCERTAIN"),
                "explanation": gemini_data.get("explanation", "Analysis complete."),
                "analysis": gemini_data.get("analysis", {}),
                "artifacts": gemini_data.get("artifacts", []),
                "timeline_data": timeline_data,
                "audio_fingerprint": {
                    "spectrogram": small_spectrogram, # 2D array
                    "frequency_bins": [], # unused
                    "amplitude_bins": [], # unused
                    "time_bins": [] # unused
                },
                "spectrogram_base64": None
            }

        except Exception as e:
            print(f"Error during analysis: {e}")
            # Fallback Mock Response on Error
            return {
                "confidence_score": 0.0,
                "verdict": "ERROR",
                "explanation": f"Analysis failed: {str(e)}",
                "analysis": {
                    "breathing": "error",
                    "prosody_variation": "error",
                    "frequency_spectrum": "error",
                    "speaking_rhythm": "error"
                },
                "artifacts": [],
                "timeline_data": [],
                "audio_fingerprint": {
                    "frequency_bins": [],
                    "amplitude_bins": [],
                    "time_bins": []
                },
                "spectrogram_base64": None 
            }
