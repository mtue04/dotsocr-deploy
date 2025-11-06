@echo off
REM Dots.OCR Quick Launch Script for Windows
REM This script activates virtual environment and runs the app

echo ========================================
echo Dots.OCR Document Processing System
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup first:
    echo   1. python -m venv venv
    echo   2. venv\Scripts\activate
    echo   3. pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if activation successful
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

echo [INFO] Virtual environment activated
echo.

REM Check if app.py exists
if not exist "app.py" (
    echo [ERROR] app.py not found!
    echo Please ensure you are in the correct directory.
    pause
    exit /b 1
)

REM Run the application
echo [INFO] Starting Dots.OCR application...
echo [INFO] This will open your browser automatically
echo [INFO] Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

python app.py

REM If app exits, show message
echo.
echo ========================================
echo Application stopped
echo ========================================
pause
