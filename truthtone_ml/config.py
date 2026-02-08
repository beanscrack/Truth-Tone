"""
TruthTone++ Configuration
All hyperparameters and paths in one place.
"""
import os
from pathlib import Path

# ── Paths ──
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
SPEC_DIR = BASE_DIR / "spectrograms"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"

# If using shared filesystem across servers, override with env var:
# export TRUTHTONE_BASE=/shared/nfs/truthtone_ml
if os.environ.get("TRUTHTONE_BASE"):
    shared = Path(os.environ["TRUTHTONE_BASE"])
    DATA_DIR = shared / "data"
    SPEC_DIR = shared / "spectrograms"
    CHECKPOINT_DIR = shared / "checkpoints"
    LOG_DIR = shared / "logs"

# ── Audio Preprocessing ──
SAMPLE_RATE = 22050          # Hz - standard for speech
AUDIO_DURATION = 4.0         # seconds to crop/pad each clip to
N_MELS = 128                 # mel frequency bins
N_FFT = 2048                 # FFT window size
HOP_LENGTH = 512             # hop between FFT windows
SPEC_IMAGE_SIZE = 224        # resize spectrogram to this (for ResNet)

# ── Dataset ──
# ASVspoof 2019 LA (Logical Access) partition
# Source: https://datashare.ed.ac.uk/handle/10283/3336
# Alternative: FakeOrReal dataset on Kaggle
ASVSPOOF_URL_BASE = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336"
DATASET_FILES = {
    "train": "LA.zip",        # ~5GB - contains train + dev + eval
}

# Labels in ASVspoof 2019
LABEL_BONAFIDE = "bonafide"   # real audio
LABEL_SPOOF = "spoof"         # fake audio
CLASS_MAP = {LABEL_BONAFIDE: 0, LABEL_SPOOF: 1}  # 0=real, 1=fake
CLASS_NAMES = ["real", "fake"]

# ── Model ──
MODEL_NAME = "resnet34"       # backbone (resnet18, resnet34, efficientnet_b0)
NUM_CLASSES = 2               # binary: real vs fake
DROPOUT = 0.4                 # dropout before final FC
FREEZE_LAYERS = True          # freeze early conv layers
UNFREEZE_FROM = "layer3"      # unfreeze from this block onward

# ── Training ──
EPOCHS = 30
BATCH_SIZE = 64               # per GPU - effective batch = BATCH_SIZE * num_gpus
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LR_SCHEDULER = "cosine"       # "cosine" or "step"
LR_STEP_SIZE = 10             # for step scheduler
LR_GAMMA = 0.1                # for step scheduler
WARMUP_EPOCHS = 2             # linear warmup
EARLY_STOP_PATIENCE = 15      # stop if val loss doesn't improve

# ── Augmentation ──
AUGMENT_TRAIN = True
AUG_TIME_MASK_MAX = 30        # SpecAugment: max time mask width
AUG_FREQ_MASK_MAX = 15        # SpecAugment: max freq mask width
AUG_NOISE_STD = 0.005         # Gaussian noise injection
AUG_MIXUP_ALPHA = 0.2         # Mixup augmentation alpha (0 = disabled)

# ── Inference ──
SEGMENT_LENGTH = 1.5          # seconds per segment for heatmap
SEGMENT_HOP = 0.5             # overlap between segments

# ── API ──
GEMINI_MODEL = "gemini-2.0-flash"
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_UPLOAD_SIZE_MB = 50
