import os
import subprocess
import sys

# Set environment variable to skip email prompt
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

# Run the streamlit app
subprocess.run([
    sys.executable, '-m', 'streamlit', 'run', 'app.py',
    '--server.port', '8501',
    '--server.headless', 'true'
])
