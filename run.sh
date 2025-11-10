#!/bin/bash

# DOTSOCR Quick Launch Script for Linux
# This script activates virtual environment and runs the app

echo "========================================"
echo "DOTSOCR Document Processing System"
echo "========================================"
echo

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo "Please run setup first:"
    echo "  1. python3 -m venv venv"
    echo "  2. source venv/bin/activate"
    echo "  3. pip install -r requirements.txt"
    echo
    echo "Or run the auto setup script:"
    echo "  chmod +x auto_setup.sh"
    echo "  ./auto_setup.sh"
    echo
    exit 1
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# Check if activation successful
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate virtual environment"
    exit 1
fi

echo "[INFO] Virtual environment activated"
echo

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "[ERROR] app.py not found!"
    echo "Please ensure you are in the correct directory."
    exit 1
fi

# Display GPU info if available
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] GPU Status:"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    echo
fi

# Run the application
echo "[INFO] Starting DOTSOCR application..."
echo "[INFO] This will open your browser automatically"
echo "[INFO] Press Ctrl+C to stop the server"
echo
echo "========================================"
echo

python app.py

# If app exits, show message
echo
echo "========================================"
echo "Application stopped"
echo "========================================"