@echo off
echo 🔍 Starting Fingerprint Forensics Toolkit...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Navigate to the project directory
cd /d "%~dp0"

REM Check if requirements are installed
echo 📦 Checking dependencies...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Dependencies not found. Installing...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

REM Run the main application
echo 🚀 Launching Fingerprint Forensics Toolkit...
python "Fingerprint Main\main.py"

pause
