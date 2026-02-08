"""
01_download_data.py
Downloads and prepares the ASVspoof 2019 LA dataset.

ASVspoof 2019 LA contains:
  - Training:  ~25,000 utterances (2,580 bonafide + 22,800 spoofed)
  - Dev:       ~25,000 utterances (2,548 bonafide + 22,296 spoofed)  
  - Eval:      ~71,000 utterances (7,355 bonafide + 63,882 spoofed)

Total: ~121,000 labeled audio files. More than enough.

If you can't download ASVspoof (it's large), see ALTERNATIVE section below.
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
from config import DATA_DIR

def download_asvspoof():
    """
    ASVspoof 2019 LA dataset.
    
    MANUAL DOWNLOAD REQUIRED (the files are behind an institutional page):
    1. Go to: https://datashare.ed.ac.uk/handle/10283/3336
    2. Download: LA.zip (~5.5 GB)
    3. Place it in: {DATA_DIR}/LA.zip
    4. Run this script to extract
    
    Alternatively, use the Zenodo mirror or academic torrent.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    zip_path = DATA_DIR / "LA.zip"
    extract_dir = DATA_DIR / "LA"
    
    if extract_dir.exists() and any(extract_dir.rglob("*.flac")):
        print(f"✓ Dataset already extracted at {extract_dir}")
        count_files(extract_dir)
        return
    
    if not zip_path.exists():
        print("=" * 60)
        print("MANUAL DOWNLOAD REQUIRED")
        print("=" * 60)
        print()
        print("ASVspoof 2019 LA dataset needs manual download:")
        print()
        print("Option A - Edinburgh DataShare (official):")
        print("  1. Visit: https://datashare.ed.ac.uk/handle/10283/3336")
        print("  2. Download LA.zip (~5.5 GB)")
        print(f"  3. Place at: {zip_path}")
        print()
        print("Option B - Direct wget (if mirror is available):")
        print(f"  wget -O {zip_path} <mirror_url>")
        print()
        print("Option C - Use the alternative dataset instead:")
        print("  python 01_download_data.py --alternative")
        print()
        print("After downloading, run this script again.")
        sys.exit(1)
    
    print(f"Extracting {zip_path}...")
    subprocess.run(["unzip", "-q", str(zip_path), "-d", str(DATA_DIR)], check=True)
    print(f"✓ Extracted to {extract_dir}")
    count_files(extract_dir)


def download_alternative():
    """
    Alternative: FakeOrReal (FOR) dataset or In-The-Wild dataset.
    Smaller, easier to download, still effective for a hackathon.
    
    FakeOrReal: ~200 files, good for quick prototyping
    In-The-Wild: ~20,000 files from various TTS systems
    
    You can also generate your own fake data using ElevenLabs API.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    alt_dir = DATA_DIR / "alternative"
    alt_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("ALTERNATIVE DATASET OPTIONS")
    print("=" * 60)
    print()
    print("Option 1 - ASVspoof 2019 LA (RECOMMENDED, best quality):")
    print("  https://datashare.ed.ac.uk/handle/10283/3336")
    print()
    print("Option 2 - In-The-Wild deepfake dataset:")
    print("  https://deepfake-demo.aisec.fraunhofer.de/in_the_wild")
    print("  ~20,000 files, modern TTS systems")
    print()
    print("Option 3 - FakeOrReal (Kaggle):")
    print("  https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset")
    print("  Smaller but quick to download")
    print()
    print("Option 4 - Generate your own with ElevenLabs:")
    print("  Use 05_generate_fakes.py to create synthetic samples")
    print("  Pair with any real speech corpus (LibriSpeech, Common Voice)")
    print()
    print("Option 5 - LibriSpeech (real only) + ElevenLabs (fakes):")
    print("  wget https://www.openslr.org/resources/12/train-clean-100.tar.gz")
    print("  This gives you 100 hours of real speech to pair with generated fakes")
    print()
    
    # Download a small real speech sample for immediate testing
    print("Downloading LibriSpeech test-clean (300MB) for immediate testing...")
    test_url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    test_tar = alt_dir / "test-clean.tar.gz"
    
    try:
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(test_tar), test_url],
            check=True
        )
        subprocess.run(
            ["tar", "-xzf", str(test_tar), "-C", str(alt_dir)],
            check=True
        )
        print(f"✓ Real speech samples extracted to {alt_dir}/LibriSpeech/test-clean")
    except subprocess.CalledProcessError:
        print("Download failed - you may need to download manually")


def setup_asvspoof_structure():
    """
    After extraction, ASVspoof 2019 LA has this structure:
    
    LA/
    ├── ASVspoof2019_LA_train/flac/     (training audio)
    ├── ASVspoof2019_LA_dev/flac/       (development/validation audio)  
    ├── ASVspoof2019_LA_eval/flac/      (evaluation audio)
    ├── ASVspoof2019_LA_cm_protocols/
    │   ├── ASVspoof2019.LA.cm.train.trn.txt   (train labels)
    │   ├── ASVspoof2019.LA.cm.dev.trl.txt     (dev labels)
    │   └── ASVspoof2019.LA.cm.eval.trl.txt    (eval labels)
    
    Protocol file format (space-separated):
    SPEAKER_ID  AUDIO_FILENAME  SYSTEM_ID  -  KEY
    LA_0079     LA_T_1138215    -          -  bonafide
    LA_0079     LA_T_5765219    A06        -  spoof
    
    KEY is either 'bonafide' (real) or 'spoof' (fake)
    """
    la_dir = DATA_DIR / "LA"
    
    if not la_dir.exists():
        print(f"LA directory not found at {la_dir}")
        print("Run download first.")
        return
    
    # Verify protocol files exist
    proto_dir = la_dir / "ASVspoof2019_LA_cm_protocols"
    if not proto_dir.exists():
        # Sometimes extracted differently
        for candidate in la_dir.rglob("*protocols*"):
            if candidate.is_dir():
                proto_dir = candidate
                break
    
    partitions = {
        "train": {
            "audio": "ASVspoof2019_LA_train/flac",
            "protocol": "ASVspoof2019.LA.cm.train.trn.txt",
        },
        "dev": {
            "audio": "ASVspoof2019_LA_dev/flac",
            "protocol": "ASVspoof2019.LA.cm.dev.trl.txt",
        },
        "eval": {
            "audio": "ASVspoof2019_LA_eval/flac", 
            "protocol": "ASVspoof2019.LA.cm.eval.trl.txt",
        },
    }
    
    for split, info in partitions.items():
        audio_dir = la_dir / info["audio"]
        proto_file = proto_dir / info["protocol"]
        
        if audio_dir.exists():
            n_files = len(list(audio_dir.glob("*.flac")))
            print(f"  {split}: {n_files} audio files")
        else:
            print(f"  {split}: MISSING audio dir at {audio_dir}")
            
        if proto_file.exists():
            with open(proto_file) as f:
                lines = f.readlines()
            n_bonafide = sum(1 for l in lines if "bonafide" in l)
            n_spoof = sum(1 for l in lines if "spoof" in l and "bonafide" not in l)
            print(f"         {n_bonafide} real + {n_spoof} fake = {len(lines)} total")
        else:
            print(f"         MISSING protocol file at {proto_file}")


def count_files(directory):
    """Count audio files in directory."""
    flac_count = len(list(Path(directory).rglob("*.flac")))
    wav_count = len(list(Path(directory).rglob("*.wav")))
    print(f"  Found: {flac_count} .flac + {wav_count} .wav files")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download TruthTone++ training data")
    parser.add_argument("--alternative", action="store_true", help="Show alternative dataset options")
    parser.add_argument("--verify", action="store_true", help="Verify existing dataset structure")
    args = parser.parse_args()
    
    if args.alternative:
        download_alternative()
    elif args.verify:
        setup_asvspoof_structure()
    else:
        download_asvspoof()
        setup_asvspoof_structure()
