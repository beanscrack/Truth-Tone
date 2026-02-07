# TruthTone++ API Contract

## Base URL
Development: `http://localhost:8000`
Production: `https://<api-url>`

## Endpoints

### 1. Analyze Audio
**POST** `/analyze`

**Request**
- `Content-Type`: `multipart/form-data`
- Body: `file` (Audio file: .wav, .mp3, .m4a)

**Response (JSON)**
```json
{
  "confidence_score": 0.87, // Float 0.0 - 1.0 (Higher = Real)
  "verdict": "REAL", // Enum: REAL, LIKELY REAL, UNCERTAIN, LIKELY FAKE, FAKE
  "explanation": "Audio shows natural human speech patterns with organic prosody variation...",
  "analysis": {
    "breathing": "natural",
    "prosody_variation": "high",
    "frequency_spectrum": "organic",
    "speaking_rhythm": "human_inconsistency"
  },
  "artifacts": [
    {
      "timestamp": 8.2,
      "type": "pitch_anomaly", 
      "severity": "low",
      "description": "Slight mechanical pitch transition"
    }
  ],
  "timeline_data": [
    {"time": 0.0, "confidence": 0.92},
    {"time": 1.0, "confidence": 0.88},
    {"time": 2.0, "confidence": 0.65}
    // ... per-second scores for heatmap
  ],
  "audio_fingerprint": {
    "frequency_bins": [/* array of values */],
    "amplitude_bins": [/* array of values */],
    "time_bins": [/* array of values */]
  },
  "spectrogram_base64": "data:image/png;base64,..."
}
```

### 2. Generate Test Fake (ElevenLabs)
**POST** `/generate-fake`

**Request**
```json
{
  "text": "This is a test of the emergency broadcast system.",
  "voice_id": "..." // Optional
}
```

**Response**
```json
{
  "audio_url": "...", 
  "analysis": { ... } // Same structure as /analyze
}
```
