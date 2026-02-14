# TruthTone++: AI Audio Authentication Platform

Winner: Best Audio AI Hack, CXC AI Hackathon 2026

TruthTone++ is a deepfake detection platform designed to restore trust in digital audio. We built this project to go beyond simple binary classification; instead of just telling you if a recording is likely fake, TruthTone++ uses 3D visualization and blockchain-backed verification to show you exactly why a piece of audio might be suspicious.

## About CxC (Compute by Create)

CxC, or Compute by Create, is the premier data science hackathon at the University of Waterloo. Organized by the Waterloo Data Science Club, it is a high-level competition where students tackle real-world industry challenges using machine learning, data analysis, and predictive modeling.

While the event attracts a community of over 2,500 young professionals annually, the physical competition is an invite-only event for a selected group of students. In 2026, the AI-focused hackathon recorded 243 participants. TruthTone++ was developed over a 36-hour sprint during this event, ultimately winning Best Audio AI Hack among a field of over 350 competing teams.

## The Problem

As AI voice cloning becomes more sophisticated, it is becoming nearly impossible for human ears to distinguish between authentic recordings and synthetic deepfakes. This technological gap has led to a rise in voice-based fraud, from $25M+ corporate narrow-casting scams to election misinformation and the general erosion of trust in audio evidence within journalism and legal settings.

## The Solution

TruthTone++ addresses this problem through a multi-layered verification system. Our platform doesn't just process sound; it analyzes the underlying fingerprints that AI models often leave behind—patterns that are invisible to the ear but detectable through forensic analysis.

### Multi-Layered AI Detection
We use a custom machine learning model trained on the ASVspoof 2019 dataset to detect generative artifacts. The system extracts audio features using librosa (including MFCCs and spectral analysis) and integrates with the Gemini API to provide plain-English explanations for its findings. Our current test datasets show a detection accuracy between 85% and 92%.

### 3D Audio Fingerprint Visualization
To make these detections intuitive, we developed a way to map frequency spectrums into interactive 3D meshes using Three.js and React Three Fiber. Human speech typically creates organic, chaotic patterns due to natural pitch variation and breathing. In contrast, AI-generated audio often reveals geometric or mathematical regularities that become obvious once you can rotate and explore the audio's "fingerprint" in 3D.

### Timeline Analysis
The platform provides a second-by-second confidence score throughout the duration of the audio. An interactive heatmap highlights specific moments where artifacts—such as impossible pitch transitions or unnatural prosody—were detected, allowing users to jump directly to suspicious segments.

### Comparison and Baseline Generation
By integrating with the ElevenLabs API, TruthTone++ can generate test deepfakes on-demand. This allows for side-by-side comparisons and visual "diffs" between original audio and AI-generated versions, helping to establish baselines for detection.

### Blockchain Verification
For audio that is verified as organic, we've integrated Solana to mint authentication certificates. By storing an immutable SHA-256 hash of the audio file on-chain, we create a timestamped record that cannot be backdated or altered, solving the trust problem for journalists and legal professionals.

## Technical Architecture

### Backend (Python/FastAPI)
The backend manages the audio processing pipeline using librosa and our custom ML classifier. It handles communication with the Gemini and ElevenLabs APIs and generates the forensic reports sent to the frontend.

### Frontend (Next.js/TypeScript)
The user interface is built with Next.js and Tailwind CSS, focusing on a clear, data-heavy dashboard. It features live audio recording, drag-and-drop uploads, and the WebGL-based visualization engine for 3D exploration.

### Blockchain (Solana)
We use Metaplex for NFT minting and metadata storage. The architecture is designed to be mainnet-ready, providing low-cost verification (typically less than $0.01 per certificate).

## Installation and Setup

### Prerequisites
- Node.js 18+
- Python 3.10+
- Google Gemini API Key
- ElevenLabs API Key (Optional)

### Quick Start

1. Clone the repository.
2. Create a .env file in the root directory based on .env.example and ensure your Gemini API key is included.
3. Run the automated setup script:
   ```bash
   ./setup_and_run.sh
   ```
   This script will set up the Python virtual environment, install all dependencies, and start both the backend (port 8000) and frontend (port 3000).

## Future Roadmap

We are currently looking into real-time detection for live calls, browser extensions for social media verification, and expanding our models to handle multi-modal analysis involving both audio and video deepfakes.

## License and Disclaimer

This project is licensed under the MIT License.

Note: This tool provides a probabilistic assessment of audio authenticity. While highly accurate, no detection system is 100% infallible. Results should be treated as forensic indicators rather than absolute proof.
