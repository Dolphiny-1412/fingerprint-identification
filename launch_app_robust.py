#!/usr/bin/env python3
"""
Robust Launcher for Fingerprint Analysis System
==============================================

This script automatically finds an available port and launches the app.
"""

import os
import sys
import subprocess
import socket

def find_free_port(start_port=8501, max_attempts=10):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None

def kill_streamlit_processes():
    """Kill any existing Streamlit processes"""
    try:
        if sys.platform == "win32":
            # Windows
            subprocess.run(["taskkill", "/f", "/im", "streamlit.exe"], 
                         capture_output=True, check=False)
            subprocess.run(["taskkill", "/f", "/im", "python.exe", "/fi", "WINDOWTITLE eq *streamlit*"], 
                         capture_output=True, check=False)
        else:
            # Unix/Linux/Mac
            subprocess.run(["pkill", "-f", "streamlit"], 
                         capture_output=True, check=False)
        print("🧹 Cleaned up existing Streamlit processes")
    except Exception as e:
        print(f"⚠️ Could not clean up processes: {e}")

def main():
    """Launch the fingerprint analysis system with robust port handling"""
    
    # Get the current directory (where this script is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    
    # Add src directory to Python path
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    print("🚀 Launching Fingerprint Analysis System...")
    print(f"📁 Source directory: {src_dir}")
    print(f"🐍 Python path updated")
    
    # Try to find a free port
    port = find_free_port()
    if port is None:
        print("❌ Could not find a free port. Trying to clean up existing processes...")
        kill_streamlit_processes()
        port = find_free_port()
        if port is None:
            print("❌ Still no free ports available. Please close other applications and try again.")
            return 1
    
    print(f"📍 Using port: {port}")
    
    # Launch the app
    app_path = os.path.join(src_dir, 'ui', 'app_modern.py')
    print(f"🎯 Starting app: {app_path}")
    
    try:
        # Launch with streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', app_path,
            '--server.port', str(port),
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
