#!/usr/bin/env python3
"""
download_model.py
Downloads the pretrained TruthTone++ model from Hugging Face Hub.

This script is called automatically by setup_and_run.sh if the model
doesn't exist locally. You can also run it manually:

    python download_model.py

The model will be saved to: truthtone_ml/checkpoints/best_model.pt
"""
import os
import sys
from pathlib import Path

# Model configuration - UPDATE THESE when you upload your model
HF_REPO_ID = "beanscrack/truthtone-model"  # e.g., "j692wu/truthtone-model"
MODEL_FILENAME = "best_model.pt"

# Local paths
SCRIPT_DIR = Path(__file__).parent
CHECKPOINT_DIR = SCRIPT_DIR / "truthtone_ml" / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / MODEL_FILENAME


def download_from_huggingface():
    """Download model from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub...")
        os.system(f"{sys.executable} -m pip install huggingface_hub -q")
        from huggingface_hub import hf_hub_download
    
    print(f"Downloading model from Hugging Face: {HF_REPO_ID}")
    print(f"This may take a few minutes for a ~250MB model...")
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=MODEL_FILENAME,
            local_dir=CHECKPOINT_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"✓ Model downloaded to: {downloaded_path}")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def download_from_url(url):
    """Fallback: download from direct URL (Google Drive, S3, etc.)."""
    import urllib.request
    
    print(f"Downloading model from: {url}")
    
    try:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, MODEL_PATH, reporthook=_progress_hook)
        print(f"\n✓ Model downloaded to: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def _progress_hook(count, block_size, total_size):
    """Show download progress."""
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rDownloading: {percent}%")
    sys.stdout.flush()


def main():
    # Check if model already exists
    if MODEL_PATH.exists():
        print(f"✓ Model already exists: {MODEL_PATH}")
        return True
    
    print("="*60)
    print("TruthTone++ Model Download")
    print("="*60)
    
    # Create checkpoint directory
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try Hugging Face first
    if HF_REPO_ID != "YOUR_USERNAME/truthtone-model":
        if download_from_huggingface():
            return True
    
    # Fallback instructions
    print("\n" + "="*60)
    print("MANUAL SETUP REQUIRED")
    print("="*60)
    print("""
To use TruthTone++, you need to either:

1. UPLOAD YOUR MODEL TO HUGGING FACE (recommended):
   - Create account at https://huggingface.co
   - Create a new model repository
   - Upload your best_model.pt file
   - Update HF_REPO_ID in this script

2. TRAIN A NEW MODEL:
   cd truthtone_ml
   python 01_download_data.py   # Download ASVspoof dataset
   python 02_preprocess.py      # Create spectrograms
   python 03_train.py           # Train the model

3. COPY EXISTING MODEL:
   If you have best_model.pt elsewhere, copy it to:
   truthtone_ml/checkpoints/best_model.pt
""")
    return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
