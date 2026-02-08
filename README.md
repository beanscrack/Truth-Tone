# TruthTone++: Advanced Deepfake Audio Detection Platform

## Overview

TruthTone++ is a multimodal AI system designed to detect synthetic and deepfake audio. It combines traditional signal processing with deep learning (CNNs on Mel Spectrograms) and Large Language Model (LLM) semantic reasoning to provide high-confidence verification of audio authenticity.

The platform analyzes audio artifacts invisible to the human ear, such as phase inconsistencies, unnatural spectral flatness, and robotic prosody, to distinguish between organic human speech and AI-generated content.

## Key Features

1.  **Hybrid Analysis Engine**: Utilizing a dual-layered approach:
    *   **Signal Analysis**: Extracting features like spectral rolloff, zero-crossing rates, and Mel-frequency cepstral coefficients (MFCCs).
    *   **Neural Network Classifier**: A ResNet-18 backbone trained on ASVSpoof datasets to detect generative artifacts in spectrogram images.
    *   **Semantic Verification**: Google Gemini 2.0 Flash integration to analyze linguistic patterns, breathing unnaturalness, and prosodic inconsistencies.

2.  **Advanced Visualization**:
    *   **3D Audio Fingerprint**: Visualization of the frequency spectrum to highlight synthetic patterns.
    *   **Timeline Heatmap**: Second-by-second confidence scoring to pinpoint specific manipulated segments.

3.  **Real-time API**: A high-performance FastAPI backend capable of processing audio files and returning detailed forensic reports in real-time.

4.  **Verification Certificates**: Generation of immutable authenticity proofs (simulated integration with Solana blockchain) for verified organic audio.

## Technology Stack

### Frontend
*   **Framework**: Next.js 14 (App Router)
*   **Styling**: Tailwind CSS
*   **Language**: TypeScript
*   **Visualization**: Custom Canvas-based renderers for spectrograms

### Backend
*   **Framework**: FastAPI (Python 3.12)
*   **ML Libraries**: PyTorch, Librosa, NumPy, Scikit-learn
*   **LLM Integration**: Google Generative AI (Gemini)
*   **Task Management**: Asyncio for non-blocking analysis

## Installation and Setup

### Prerequisites
*   Node.js 18+
*   Python 3.10+
*   Google Gemini API Key
*   ElevenLabs API Key (Optional, for generating test fakes)

### Quick Start

1.  Clone the repository.
2.  Configure environment variables:
    Create a `.env` file in the root directory based on `.env.example`.
    Ensure `GEMINI_API_KEY` is set.
3.  Run the automated setup script:

    ```bash
    ./setup_and_run.sh
    ```

    This script will:
    *   Create a Python virtual environment and install dependencies.
    *   Install frontend NPM packages.
    *   Start the FastAPI backend on port 8000.
    *   Start the Next.js frontend on port 3000.

4.  Access the application at `http://localhost:3000`.

## API Documentation

### POST /api/analyze

Analyzes an uploaded audio file.

**Request:**
*   Content-Type: `multipart/form-data`
*   Body: `file` (Binary audio file: WAV, MP3, M4A)

**Response:**
Returns a JSON object containing the `confidence_score` (0.0 to 1.0), `verdict` (REAL/FAKE), and detailed `analysis` metrics.

### POST /api/generate-fake

Generates synthetic audio for testing purposes using ElevenLabs.

**Request:**
*   Content-Type: `application/json`
*   Body: `{ "text": "Text to synthesize" }`

**Response:**
Returns the generated audio filename and its immediate analysis results.

## Model Training

The ML model is located in the `truthtone_ml` directory. It uses a transfer learning approach, fine-tuning a pre-trained ResNet-18 on Mel Spectrogram images generated from the ASVSpoof 2019 dataset.

## License

MIT License.

## Disclaimer

This tool provides a probabilistic assessment of audio authenticity. While highly accurate, no detection system is 100% fallacy-proof. Results should be used as forensic indicators rather than absolute proof.
