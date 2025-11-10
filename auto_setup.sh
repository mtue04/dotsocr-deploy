#!/bin/bash

# DOTSOCR Auto Setup Script for Linux GPU Servers
# This script automatically clones, sets up and runs DOTSOCR on a fresh Linux server

set -e  # Exit on any error

echo "========================================"
echo "DOTSOCR Auto Setup for Linux GPU Server"
echo "========================================"
echo

# Update system packages
echo "[1/8] Updating system packages..."
sudo apt update && sudo apt upgrade -y
echo "[OK] System updated"
echo

# Install essential packages
echo "[2/8] Installing essential packages..."
sudo apt install -y git python3 python3-pip python3-venv curl wget build-essential
echo "[OK] Essential packages installed"
echo

# Check for NVIDIA drivers and CUDA
echo "[3/8] Checking GPU and CUDA installation..."
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
    echo "[OK] GPU drivers found"
else
    echo "[WARNING] NVIDIA drivers not found. Installing..."
    # Install NVIDIA drivers
    sudo apt install -y nvidia-driver-535
    echo "[INFO] NVIDIA drivers installed. You may need to reboot after setup."
fi

# Install CUDA if not present
if ! command -v nvcc >/dev/null 2>&1; then
    echo "[INFO] Installing CUDA toolkit..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2004-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-6
    rm cuda-repo-ubuntu2004-12-6-local_12.6.3-560.35.05-1_amd64.deb
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
fi
echo

# Clone the repository
echo "[4/8] Cloning DOTSOCR repository..."
if [ -d "dotsocr-deploy" ]; then
    echo "[WARNING] dotsocr-deploy directory already exists"
    read -p "Do you want to remove it and clone fresh? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf dotsocr-deploy
        git clone https://github.com/mtue04/dotsocr-deploy.git
    fi
else
    git clone https://github.com/mtue04/dotsocr-deploy.git
fi
cd dotsocr-deploy
echo "[OK] Repository cloned and entered directory"
echo

# Create virtual environment
echo "[5/8] Creating Python virtual environment..."
if [ -d "venv" ]; then
    echo "[WARNING] Virtual environment already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
    fi
else
    python3 -m venv venv
fi
source venv/bin/activate
echo "[OK] Virtual environment created and activated"
echo

# Upgrade pip
echo "[6/8] Upgrading pip..."
python -m pip install --upgrade pip
echo

# Install PyTorch with CUDA support
echo "[7/8] Installing PyTorch with CUDA support..."
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] Installing GPU-enabled PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "[INFO] Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
fi
echo

# Install other dependencies
echo "[8/8] Installing other dependencies..."
echo "This may take 10-30 minutes..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo
    echo "[ERROR] Some packages failed to install"
    echo "Common issues:"
    echo "  - flash-attn: Try running: pip install flash-attn --no-build-isolation"
    echo "  - Or skip it and use eager attention (see README)"
    echo
    exit 1
fi

echo
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo
echo "Your DOTSOCR is now ready to use!"
echo
echo "To start the application:"
echo "  ./run.sh"
echo
echo "To manually activate and run:"
echo "  cd dotsocr-deploy"
echo "  source venv/bin/activate"
echo "  python app.py"
echo
echo "First run will download the model (~8GB)"
echo "This may take 10-30 minutes depending on your internet speed"
echo
echo "For more information, see:"
echo "  - README.md (full documentation)"
echo "  - QUICKSTART.md (quick guide)"
echo

# Ask if user wants to start immediately
read -p "Do you want to start the application now? (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "Starting DOTSOCR application..."
    python app.py
fi