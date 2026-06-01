#!/usr/bin/env python3
"""
Simple Launcher for Fingerprint Analysis System
==============================================

This script provides an easy way to run the application
with proper path configuration.
"""

import sys
import os
import subprocess

def main():
    """Launch the fingerprint analysis system"""
    
    # Add src directory to Python path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    sys.path.insert(0, src_path)
    
    print("🚀 Launching Fingerprint Analysis System...")
    print(f"📁 Source directory: {src_path}")
    
    # Launch with streamlit directly
    try:
        app_path = os.path.join(src_path, 'ui', 'app_modern.py')
        print(f"🔄 Launching with Streamlit: {app_path}")
        
        # Launch with streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', app_path,
            '--server.port', '8501',
            '--server.address', 'localhost'
        ])
        
    except Exception as e:
        print(f"❌ Failed to launch: {e}")
        print("\n🔧 Manual launch instructions:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run: streamlit run src/ui/app_modern.py")
        print("3. Or run: python main.py")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
