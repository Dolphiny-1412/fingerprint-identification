#!/usr/bin/env python3
"""
Main Entry Point for Fingerprint Analysis System
===============================================

This is the main entry point that launches the Streamlit application.
All core functionality has been organized into the src/ directory structure.
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main application
from ui.app_modern import main

if __name__ == "__main__":
    main()
