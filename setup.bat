@echo off
REM Dots.OCR Setup Script for Windows
REM This script sets up the environment and installs dependencies

echo ========================================
echo Dots.OCR - Setup Script
echo ========================================
echo.

REM Check Python installation
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

python --version
echo [OK] Python is installed
echo.

REM Create virtual environment
echo [2/5] Creating virtual environment...
if exist "venv" (
    echo [WARNING] Virtual environment already exists
    choice /C YN /M "Do you want to recreate it"
    if errorlevel 2 goto skip_venv
    echo Removing old virtual environment...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment created

:skip_venv
echo.

REM Activate virtual environment
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo [4/5] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip, continuing anyway...
)
echo.

REM Install dependencies
echo [5/5] Installing dependencies...
echo This may take 10-30 minutes...
echo.

REM Check for CUDA
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [INFO] CUDA not detected, installing CPU-only PyTorch...
    pip install torch torchvision torchaudio
) else (
    echo [INFO] CUDA detected, installing GPU-enabled PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
)

echo.
echo [INFO] Installing other dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Some packages failed to install
    echo Common issues:
    echo   - flash-attn: Try running: pip install flash-attn --no-build-isolation
    echo   - Or skip it and use eager attention (see README)
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo Next steps:
echo   1. Run: run.bat
echo   2. Or manually: venv\Scripts\activate.bat
echo                   python app.py
echo.
echo First run will download the model (~8GB)
echo This may take 10-30 minutes depending on your internet speed
echo.
echo For more information, see:
echo   - README.md (full documentation)
echo   - QUICKSTART.md (quick guide)
echo.
pause
