#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== TruthTone++ Setup & Run Script ===${NC}"

# 1. Backend Setup
echo -e "\n${GREEN}[1/4] Setting up Python Backend...${NC}"
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Using active Conda environment: $CONDA_DEFAULT_ENV"
fi

echo "Installing Python dependencies..."
pip install -r requirements.txt

# 2. Frontend Setup
echo -e "\n${GREEN}[2/4] Setting up Frontend...${NC}"
cd frontend
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
else
    echo "Node modules already installed."
fi
cd ..

# 3. Start Servers
echo -e "\n${GREEN}[3/4] Starting Servers...${NC}"

# Kill any existing processes on these ports
pkill -f "uvicorn backend.server:app"
pkill -f "next-server"
pkill -f "python 05_api.py"

# Start Backend in background
echo "Starting Backend (Port 8000)..."

# Check if model exists, download if not
if [ ! -f "truthtone_ml/checkpoints/best_model.pt" ]; then
    echo "Model not found. Attempting to download..."
    python download_model.py
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERROR: Model not available. See instructions above.${NC}"
        exit 1
    fi
fi

cd truthtone_ml
python 05_api.py --model checkpoints/best_model.pt --port 8000 --host 0.0.0.0 &
cd ..
BACKEND_PID=$!

# Start Frontend
echo "Starting Frontend (Port 3000)..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo -e "\n${GREEN}[4/4] Setup Complete!${NC}"
echo -e "${BLUE}Backend running at: http://localhost:8000${NC}"
echo -e "${BLUE}Frontend running at: http://localhost:3000${NC}"
echo -e "\nPress CTRL+C to stop all servers."

# Trap SIGINT to kill both processes when script is stopped
trap "kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT

# Wait for processes
wait
