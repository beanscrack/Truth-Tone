"""
02_preprocess.py
Converts raw audio files into mel spectrogram images for training.

This is the most important preprocessing step. The quality of your
spectrograms directly determines model accuracy.

Usage:
    python 02_preprocess.py                    # default 8 workers
    python 02_preprocess.py --workers 16       # faster on multi-core
    python 02_preprocess.py --verify           # check existing spectrograms
"""
import os
import json
import argparse
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from PIL import Image

from config import (
    DATA_DIR, SPEC_DIR, SAMPLE_RATE, AUDIO_DURATION,
    N_MELS, N_FFT, HOP_LENGTH, SPEC_IMAGE_SIZE,
    LABEL_BONAFIDE, LABEL_SPOOF, CLASS_MAP
)


def parse_protocol_file(protocol_path):
    """
    Parse ASVspoof 2019 protocol file.
    
    Format: SPEAKER_ID  AUDIO_FILENAME  SYSTEM_ID  -  KEY
    Returns: list of (filename, label) tuples
    """
    entries = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                filename = parts[1]      # e.g., LA_T_1138215
                label = parts[4]         # bonafide or spoof
                entries.append((filename, label))
    return entries


def audio_to_spectrogram(audio_path, sr=SAMPLE_RATE, duration=AUDIO_DURATION):
    """
    Convert audio file to mel spectrogram.
    
    Returns:
        mel_db: numpy array of shape (n_mels, time_frames) in dB scale
        raw_audio: the loaded audio waveform (for frequency extraction later)
    """
    # Load audio (handles .flac, .wav, .mp3)
    try:
        y, orig_sr = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return None, None
    
    # Pad or trim to exact duration
    target_length = int(sr * duration)
    if len(y) < target_length:
        # Pad with zeros
        y = np.pad(y, (0, target_length - len(y)), mode="constant")
    else:
        y = y[:target_length]
    
    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    
    # Convert to dB scale (log scale - makes patterns more visible)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    return mel_db, y


def spectrogram_to_image(mel_db, size=SPEC_IMAGE_SIZE):
    """
    Convert mel spectrogram to a normalized image array.
    
    Normalizes to [0, 255] uint8 for storage efficiency,
    then resize to target size for the CNN.
    """
    # Normalize to 0-255
    mel_norm = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8) * 255)
    mel_uint8 = mel_norm.astype(np.uint8)
    
    # Resize to CNN input size
    img = Image.fromarray(mel_uint8)
    img = img.resize((size, size), Image.BILINEAR)
    
    return np.array(img)


def process_single_file(args, audio_dir, output_dir):
    """Process a single audio file. Used by multiprocessing pool."""
    filename, label = args
    
    # Find the audio file (try .flac first, then .wav)
    audio_path = audio_dir / f"{filename}.flac"
    if not audio_path.exists():
        audio_path = audio_dir / f"{filename}.wav"
    if not audio_path.exists():
        return None
    
    # Output path
    label_dir = output_dir / ("real" if label == LABEL_BONAFIDE else "fake")
    output_path = label_dir / f"{filename}.npy"
    
    # Skip if already processed
    if output_path.exists():
        return str(output_path)
    
    # Convert to spectrogram
    mel_db, _ = audio_to_spectrogram(audio_path)
    if mel_db is None:
        return None
    
    # Convert to image array
    spec_img = spectrogram_to_image(mel_db)
    
    # Save as numpy array (much faster to load than PNG during training)
    label_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_path, spec_img)
    
    return str(output_path)


def preprocess_split(split_name, audio_dir, protocol_path, output_dir, workers=8):
    """Preprocess all files in one split (train/dev/eval)."""
    print(f"\n{'='*60}")
    print(f"Processing {split_name}")
    print(f"{'='*60}")
    
    # Parse protocol
    entries = parse_protocol_file(protocol_path)
    print(f"  Found {len(entries)} entries in protocol file")
    
    n_real = sum(1 for _, l in entries if l == LABEL_BONAFIDE)
    n_fake = sum(1 for _, l in entries if l == LABEL_SPOOF)
    print(f"  Real: {n_real}, Fake: {n_fake}")
    
    # Create output dirs
    (output_dir / "real").mkdir(parents=True, exist_ok=True)
    (output_dir / "fake").mkdir(parents=True, exist_ok=True)
    
    # Process in parallel
    process_fn = partial(process_single_file, audio_dir=audio_dir, output_dir=output_dir)
    
    results = []
    with Pool(workers) as pool:
        for result in tqdm(
            pool.imap_unordered(process_fn, entries),
            total=len(entries),
            desc=f"  {split_name}"
        ):
            if result:
                results.append(result)
    
    print(f"  ✓ Processed {len(results)}/{len(entries)} files")
    
    # Save manifest
    manifest = {
        "split": split_name,
        "total": len(results),
        "real": len([r for r in results if "/real/" in r]),
        "fake": len([r for r in results if "/fake/" in r]),
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    
    return manifest


def preprocess_asvspoof(workers=8):
    """Process full ASVspoof 2019 LA dataset."""
    la_dir = DATA_DIR / "LA"
    proto_dir = None
    
    # Find protocol directory (structure can vary after extraction)
    for candidate in [
        la_dir / "ASVspoof2019_LA_cm_protocols",
        la_dir / "LA" / "ASVspoof2019_LA_cm_protocols",
    ]:
        if candidate.exists():
            proto_dir = candidate
            break
    
    if not proto_dir:
        # Search recursively
        for p in la_dir.rglob("*cm_protocols*"):
            if p.is_dir():
                proto_dir = p
                break
    
    if not proto_dir:
        print("ERROR: Cannot find protocol files directory.")
        print(f"Searched in: {la_dir}")
        print("Please check your dataset extraction.")
        return
    
    splits = {
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
    
    all_manifests = {}
    
    for split_name, info in splits.items():
        # Find audio directory
        audio_dir = la_dir / info["audio"]
        if not audio_dir.exists():
            # Try alternate path
            audio_dir = la_dir / "LA" / info["audio"]
        if not audio_dir.exists():
            print(f"  WARNING: Audio dir not found for {split_name}: {audio_dir}")
            continue
        
        protocol_path = proto_dir / info["protocol"]
        if not protocol_path.exists():
            print(f"  WARNING: Protocol file not found: {protocol_path}")
            continue
        
        output_dir = SPEC_DIR / split_name
        manifest = preprocess_split(
            split_name, audio_dir, protocol_path, output_dir, workers
        )
        all_manifests[split_name] = manifest
    
    # Summary
    print(f"\n{'='*60}")
    print("PREPROCESSING COMPLETE")
    print(f"{'='*60}")
    for split, m in all_manifests.items():
        print(f"  {split}: {m['real']} real + {m['fake']} fake = {m['total']} total")
    print(f"\nSpectrograms saved to: {SPEC_DIR}")


def preprocess_custom_directory(audio_dir, workers=8):
    """
    Process a custom directory with structure:
    audio_dir/
    ├── real/     (real audio files)
    └── fake/     (fake audio files)
    """
    audio_dir = Path(audio_dir)
    output_dir = SPEC_DIR / "custom"
    
    for label in ["real", "fake"]:
        label_dir = audio_dir / label
        if not label_dir.exists():
            print(f"  WARNING: {label_dir} not found")
            continue
        
        files = list(label_dir.glob("*.*"))
        audio_files = [f for f in files if f.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".ogg"}]
        print(f"  Found {len(audio_files)} {label} audio files")
        
        asvspoof_label = LABEL_BONAFIDE if label == "real" else LABEL_SPOOF
        entries = [(f.stem, asvspoof_label) for f in audio_files]
        
        (output_dir / label).mkdir(parents=True, exist_ok=True)
        
        process_fn = partial(process_single_file, audio_dir=label_dir, output_dir=output_dir)
        
        with Pool(workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_fn, entries),
                total=len(entries),
                desc=f"  {label}"
            ))


def verify_spectrograms():
    """Verify preprocessed spectrograms are valid."""
    print("Verifying spectrograms...")
    
    for split_dir in SPEC_DIR.iterdir():
        if not split_dir.is_dir():
            continue
        
        for label in ["real", "fake"]:
            label_dir = split_dir / label
            if not label_dir.exists():
                continue
            
            files = list(label_dir.glob("*.npy"))
            
            # Check a sample
            bad = 0
            for f in files[:100]:
                try:
                    arr = np.load(f)
                    assert arr.shape == (SPEC_IMAGE_SIZE, SPEC_IMAGE_SIZE), f"Bad shape: {arr.shape}"
                    assert arr.dtype == np.uint8, f"Bad dtype: {arr.dtype}"
                except Exception as e:
                    print(f"  BAD: {f}: {e}")
                    bad += 1
            
            status = "✓" if bad == 0 else f"✗ ({bad} bad)"
            print(f"  {split_dir.name}/{label}: {len(files)} files {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio to spectrograms")
    parser.add_argument("--workers", type=int, default=min(8, cpu_count()),
                        help="Number of parallel workers")
    parser.add_argument("--custom-dir", type=str, default=None,
                        help="Path to custom audio directory (real/ and fake/ subdirs)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing spectrograms")
    args = parser.parse_args()
    
    if args.verify:
        verify_spectrograms()
    elif args.custom_dir:
        preprocess_custom_directory(args.custom_dir, args.workers)
    else:
        preprocess_asvspoof(args.workers)
