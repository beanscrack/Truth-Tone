# <p align="center">TruthTone++</p>

<p align="center">
  <strong>Restoring Trust in Digital Audio through Forensic AI Analysis & Blockchain Verification</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Winner-Best%20Audio%20AI%20Hack-gold?style=for-the-badge&logo=target" alt="Winner Best Audio AI Hack">
  <img src="https://img.shields.io/badge/Hackathon-CxC%202026-blue?style=for-the-badge" alt="CxC 2026">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License MIT">
</p>

---

## Overview

**TruthTone++** is a state-of-the-art deepfake detection platform designed to rebuild trust in the digital age. Beyond simple binary classification, TruthTone++ provides detailed forensic insights using **3D spectral visualization**, **timeline confidence mapping**, and **blockchain-backed authentication**.

Developed in a 36-hour sprint at **CxC (Compute by Create) 2026**, the premier data science hackathon at the University of Waterloo, TruthTone++ beat out over 350 teams to win **Best Audio AI Hack**.

---

## Features

- **Multi-Layered AI Detection**: Custom ML models trained on the ASVspoof 2019 dataset extract MFCCs and spectral features to detect generative artifacts with 85%–92% accuracy.
- **3D Audio Fingerprinting**: Revolutionary interactive 3D meshes (built with Three.js) that reveal the geometric regularities of AI-generated sound vs. the organic chaos of human speech.
- **Timeline Analysis**: A second-by-second confidence heatmap highlighting precisely where an audio clip might have been tampered with.
- **Solana-Backed Verification**: Immutable certificate minting on the Solana blockchain (via Metaplex) to create permanent, tamper-proof records of verified organic audio.
- **Explanatory AI**: Integration with Google Gemini to provide human-readable forensic reports, translating complex signal analysis into plain English.
- **Deepfake Comparison**: Direct integration with ElevenLabs for generating and comparing "visual diffs" between original and synthetic audio.

---

## Technical Architecture

### Backend
- **Framework**: Python / FastAPI
- **Signal Processing**: Librosa for spectral analysis and feature extraction.
- **AI Models**: Custom ML classifier + Google Gemini API for natural language analysis.
- **Generation**: ElevenLabs API for baseline deepfake creation.

### Frontend
- **Framework**: Next.js (TypeScript) + Tailwind CSS.
- **Visualization**: Three.js & React Three Fiber for WebGL-based 3D spectral rendering.
- **User Experience**: Real-time audio recording and drag-and-drop dashboard.

### Blockchain
- **Network**: Solana
- **Protocol**: Metaplex for NFT-based authentication certificates.
- **Security**: SHA-256 file hashing for immutable on-chain records.

---

## Quick Start

### Prerequisites
- **Node.js**: 18.x or higher
- **Python**: 3.10 or higher
- **API Keys**: Google Gemini & ElevenLabs (Optional for comparison feature)

### Installation

1. **Clone the project**
   ```bash
   git clone https://github.com/beanscrack/Truth-Tone.git
   cd Truth-Tone
   ```

2. **Configure Environment**
   Create a `.env` file in the root directory:
   ```bash
   cp .env.example .env
   # Add your GEMINI_API_KEY to the .env file
   ```

3. **Launch the Platform**
   TruthTone++ includes an automated setup script that handles the venv, dependencies, and starting both servers:
   ```bash
   chmod +x setup_and_run.sh
   ./setup_and_run.sh
   ```

*The backend will run on `localhost:8000` and the frontend on `localhost:3000`.*

---

## Future Roadmap

- [ ] **Live Call Detection**: Real-time analysis for VoIP and phone calls.
- [ ] **Browser Extension**: Instant verification for social media and news sites.
- [ ] **Multi-Modal Support**: Expanding detection to video deepfakes.
- [ ] **Enterprise API**: Plug-and-play forensic auditing for legal and journalistic firms.

---

## Disclaimer

*TruthTone++ provides a probabilistic assessment of audio authenticity. While highly accurate, no detection system is 100% infallible. Results should be treated as forensic indicators rather than absolute proof.*
