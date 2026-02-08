#!/bin/bash
# run_backend.sh - Run only the ML inference API (for GPU servers without Node.js)

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=== TruthTone++ Backend Server ===${NC}"

# Check for model
if [ ! -f "truthtone_ml/checkpoints/best_model.pt" ]; then
    echo -e "${RED}Model not found at truthtone_ml/checkpoints/best_model.pt${NC}"
    echo "Run: python download_model.py"
    exit 1
fi

# Kill existing
pkill -f "python 05_api.py" 2>/dev/null

echo -e "${GREEN}Starting ML API on port 8000...${NC}"
cd truthtone_ml
python 05_api.py --model checkpoints/best_model.pt --port 8000 --host 0.0.0.0

