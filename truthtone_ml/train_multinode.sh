#!/bin/bash
# ═══════════════════════════════════════════════════════════
# Multi-Node Training: 2 Servers × 1 RTX 6000 Pro each
# ═══════════════════════════════════════════════════════════
#
# SETUP:
#   - Server 1 (master): Has 1x RTX 6000 Pro
#   - Server 2 (worker): Has 1x RTX 6000 Pro
#   - Shared filesystem (NFS/CIFS) mounted at same path on both
#
# BEFORE RUNNING:
#   1. Edit SERVER1_IP below with your master server's IP
#   2. Make sure both servers can reach each other on port 29500
#   3. Make sure the shared filesystem has the data + code
#   4. Install dependencies on BOTH servers
#
# USAGE:
#   On Server 1: bash train_multinode.sh 0
#   On Server 2: bash train_multinode.sh 1
# ═══════════════════════════════════════════════════════════

set -e

# ── EDIT THESE ──
SERVER1_IP="10.0.0.1"          # <-- CHANGE to your master server IP
MASTER_PORT=29500               # Free port for NCCL communication
SHARED_BASE="/shared/truthtone_ml"  # <-- CHANGE to your shared filesystem path

# Or use the local directory if both servers see the same mount
# SHARED_BASE="$(pwd)"

# ── Get node rank from argument ──
NODE_RANK=${1:-0}

if [ -z "$1" ]; then
    echo "Usage: bash train_multinode.sh <node_rank>"
    echo "  node_rank 0 = master (run on Server 1)"
    echo "  node_rank 1 = worker (run on Server 2)"
    exit 1
fi

echo "═══════════════════════════════════════════════════════"
echo " TruthTone++ Multi-Node Training"
echo " Node Rank: $NODE_RANK"
echo " Master: $SERVER1_IP:$MASTER_PORT"
echo "═══════════════════════════════════════════════════════"

# Use shared filesystem for data
export TRUTHTONE_BASE="$SHARED_BASE"

# Activate environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Verify GPU
echo ""
python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_mem/1e9:.0f}GB')
"

# ── NCCL settings for multi-node ──
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0    # <-- CHANGE if your network interface differs (check with `ip addr`)
# export NCCL_IB_DISABLE=1        # Uncomment if InfiniBand issues

echo ""
echo "Starting distributed training..."
echo ""

torchrun \
    --nnodes=2 \
    --node_rank=$NODE_RANK \
    --nproc_per_node=1 \
    --master_addr=$SERVER1_IP \
    --master_port=$MASTER_PORT \
    03_train.py \
        --epochs 30 \
        --batch-size 64 \
        --lr 3e-4 \
        --distributed \
        --model resnet18 \
        --workers 8 \
        --unfreeze-epoch 5

echo ""
echo "Training complete on node $NODE_RANK"
