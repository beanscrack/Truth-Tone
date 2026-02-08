#!/bin/bash
# ═══════════════════════════════════════════════════════════
# TruthTone++ Quick Start
# Run this on your Linux server with RTX 6000 Pro
# ═══════════════════════════════════════════════════════════

set -e

echo "═══════════════════════════════════════════════════════"
echo " TruthTone++ ML Pipeline Setup"
echo "═══════════════════════════════════════════════════════"

# ── Step 1: Environment ──
echo ""
echo "[1/6] Setting up Python environment..."

# Create venv (recommended)
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# Verify GPU
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem/1e9:.0f}GB)')
"

# ── Step 2: Dataset ──
echo ""
echo "[2/6] Dataset setup..."
echo ""
echo "You need the ASVspoof 2019 LA dataset."
echo "Download from: https://datashare.ed.ac.uk/handle/10283/3336"
echo "Place LA.zip in ./data/ then run:"
echo "  python3 01_download_data.py"
echo ""

if [ -f "data/LA.zip" ]; then
    python3 01_download_data.py
elif [ -d "data/LA" ]; then
    python3 01_download_data.py --verify
else
    echo "⚠ Dataset not found. Download it first."
    echo "  Run: python3 01_download_data.py  (for instructions)"
fi

# ── Step 3: Preprocess ──
echo ""
echo "[3/6] Preprocessing audio → spectrograms..."

if [ -d "spectrograms/train" ]; then
    echo "  Spectrograms already exist. Verifying..."
    python3 02_preprocess.py --verify
else
    if [ -d "data/LA" ]; then
        # Use all CPU cores for preprocessing
        NPROC=$(nproc)
        echo "  Using $NPROC workers..."
        python3 02_preprocess.py --workers $NPROC
    else
        echo "  ⚠ Skipping - no dataset found"
    fi
fi

# ── Step 4: Train ──
echo ""
echo "[4/6] Training model..."

if [ -d "spectrograms/train" ]; then
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    
    if [ "$NUM_GPUS" -gt 1 ]; then
        echo "  Detected $NUM_GPUS GPUs - using distributed training"
        torchrun --nproc_per_node=$NUM_GPUS 03_train.py \
            --epochs 30 \
            --batch-size 64 \
            --lr 3e-4 \
            --distributed \
            --model resnet18 \
            --workers 8 \
            --unfreeze-epoch 5
    elif [ "$NUM_GPUS" -eq 1 ]; then
        echo "  Detected 1 GPU - single GPU training"
        python3 03_train.py \
            --epochs 30 \
            --batch-size 64 \
            --lr 3e-4 \
            --model resnet18 \
            --workers 8 \
            --unfreeze-epoch 5
    else
        echo "  ⚠ No GPU detected - training on CPU (will be slow)"
        python3 03_train.py --epochs 10 --batch-size 16 --workers 4
    fi
else
    echo "  ⚠ Skipping - no spectrograms found"
fi

# ── Step 5: Evaluate ──
echo ""
echo "[5/6] Evaluating model..."

if [ -f "checkpoints/best_model.pt" ]; then
    python3 04_evaluate.py --model checkpoints/best_model.pt --split eval
    echo ""
    echo "  Results saved to checkpoints/eval_results/"
else
    echo "  ⚠ Skipping - no trained model found"
fi

# ── Step 6: Start API ──
echo ""
echo "[6/6] Ready to start API server"

if [ -f "checkpoints/best_model.pt" ]; then
    echo ""
    echo "  To start the inference API:"
    echo "    python3 05_api.py --model checkpoints/best_model.pt --port 8000"
    echo ""
    echo "  Set these environment variables first:"
    echo "    export GEMINI_API_KEY=your_key_here"
    echo "    export ELEVENLABS_API_KEY=your_key_here"
    echo ""
    echo "  API docs will be at: http://localhost:8000/docs"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo " Setup complete!"
echo "═══════════════════════════════════════════════════════"
