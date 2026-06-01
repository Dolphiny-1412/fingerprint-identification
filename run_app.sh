#!/bin/bash

echo "🚀 Launching Fingerprint Analysis System..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check if requirements are installed
echo "📦 Checking dependencies..."
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "⚠️ Streamlit not found. Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

echo "✅ Dependencies ready!"
echo "🎯 Launching application..."

# Launch the application
echo "🎯 Starting Streamlit server..."
streamlit run src/ui/app_modern.py --server.port 8501 --server.address localhost

if [ $? -ne 0 ]; then
    echo
    echo "❌ Application failed to launch"
    echo "🔧 Try running manually: streamlit run src/ui/app_modern.py"
fi

echo
echo "👋 Application closed"
