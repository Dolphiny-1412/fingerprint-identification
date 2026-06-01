#!/usr/bin/env python3
"""
Simple Launcher for Fingerprint Analysis System
==============================================

This script ensures proper path configuration and launches the app.
"""

import os
import sys
import subprocess

def main():
    """Launch the fingerprint analysis system with proper path setup"""
    
    # Get the current directory (where this script is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    
    # Add src directory to Python path
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    print("🚀 Launching Fingerprint Analysis System...")
    print(f"📁 Source directory: {src_dir}")
    print(f"🐍 Python path updated")
    
    # Launch the app
    app_path = os.path.join(src_dir, 'ui', 'app_modern.py')
    print(f"🎯 Starting app: {app_path}")
    
    try:
        # Launch with streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', app_path,
            '--server.port', '8502',  # Changed from 8501 to 8502
            '--server.address', 'localhost'
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
