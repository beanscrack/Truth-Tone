# TruthTone++ ML Training Pipeline

## Hardware Requirements
- 2x NVIDIA RTX 6000 Pro (shared filesystem between servers)
- ~50GB disk space for dataset + spectrograms + models

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset
python 01_download_data.py

# 3. Preprocess audio → mel spectrograms
python 02_preprocess.py --workers 16

# 4. Train model (single GPU)
python 03_train.py --epochs 30 --batch-size 64

# OR: Train distributed across 2 GPUs (same machine)
torchrun --nproc_per_node=2 03_train.py --epochs 30 --batch-size 64 --distributed

# OR: Train distributed across 2 servers (shared filesystem)
# On server 1:
torchrun --nnodes=2 --node_rank=0 --master_addr=SERVER1_IP --master_port=29500 --nproc_per_node=1 03_train.py --epochs 30 --batch-size 64 --distributed
# On server 2:
torchrun --nnodes=2 --node_rank=1 --master_addr=SERVER1_IP --master_port=29500 --nproc_per_node=1 03_train.py --epochs 30 --batch-size 64 --distributed

# 5. Evaluate
python 04_evaluate.py --model checkpoints/best_model.pt

# 6. Run inference API
python 05_api.py --model checkpoints/best_model.pt --port 8000
```

## Project Structure
```
truthtone_ml/
├── README.md
├── requirements.txt
├── 01_download_data.py      # Dataset download + verification
├── 02_preprocess.py         # Audio → mel spectrogram conversion
├── 03_train.py              # Model training (single + multi-GPU)
├── 04_evaluate.py           # Evaluation + confusion matrix + metrics
├── 05_api.py                # FastAPI inference server
├── model.py                 # Model architecture definition
├── dataset.py               # PyTorch dataset class
├── config.py                # All hyperparameters in one place
├── data/                    # Raw audio files (created by download script)
├── spectrograms/            # Preprocessed spectrograms (created by preprocess)
└── checkpoints/             # Saved models (created during training)
```
