@echo off
echo 🚀 Launching Fingerprint Analysis System...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if requirements are installed
echo 📦 Checking dependencies...
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Streamlit not found. Installing dependencies...
    pip install streamlit
    if errorlevel 1 (
        echo ❌ Failed to install Streamlit
        echo 🔧 Trying to install all requirements...
        pip install -r requirements.txt
        if errorlevel 1 (
            echo ❌ Failed to install dependencies
            pause
            exit /b 1
        )
    )
)

REM Check if streamlit command is available
streamlit --version >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Streamlit command not found in PATH
    echo 🔧 Trying to use python -m streamlit...
    python -m streamlit --version >nul 2>&1
    if errorlevel 1 (
        echo ❌ Streamlit not properly installed
        echo 🔧 Please install manually: pip install streamlit
        pause
        exit /b 1
    )
    set STREAMLIT_CMD=python -m streamlit
) else (
    set STREAMLIT_CMD=streamlit
)

echo ✅ Dependencies ready!
echo 🎯 Launching application...

REM Launch the application
echo 🎯 Starting Streamlit server...
echo 🔧 Using robust launcher to find available port...
python launch_app_robust.py

if errorlevel 1 (
    echo.
    echo ❌ Application failed to launch
    echo 🔧 Try running manually: python -m streamlit run src/ui/app_modern.py
    pause
)

echo.
echo 👋 Application closed
pause
