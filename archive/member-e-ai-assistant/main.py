# main.py

import argparse
import json
import os
from agent.assistant import AIAssistant

def load_input_data(file_path):
    """
    Load fingerprint data from a JSON file. If failed, return None.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded input data from {file_path}.")
        return data
    except Exception as e:
        print(f"Failed to load input file {file_path}: {e}")
        return None

def normalize_input(data):
    """
    Normalize input data formats (e.g., string numbers to ints, pattern case).
    """
    normalized = {}
    for key, value in data.items():
        if key.lower() in ['minutiae_count', 'minutiae']:
            # Convert numeric strings to int
            try:
                normalized[key] = int(value)
            except Exception:
                normalized[key] = value
        elif key.lower() == 'pattern':
            # Normalize pattern string (lowercase)
            normalized[key] = str(value).lower()
        else:
            normalized[key] = value
    return normalized

def main():
    parser = argparse.ArgumentParser(description="Fingerprint Forensics Toolkit - AIAssistant Demo")
    parser.add_argument("--file", type=str, help="Path to fingerprint data JSON file (required unless you use another entrypoint)")
    parser.add_argument("--explain", action="store_true", help="Enable AI explanation of results")
    args = parser.parse_args()

    if not args.file:
        print("ERROR: --file is required for main.py (this CLI expects a single JSON payload).")
        print("Hint: use app.py (streamlit) for catalog-based browsing, or pass --file path/to/sample.json")
        return

    input_data = load_input_data(args.file)
    if not input_data:
        print("Failed to load input data, aborting.")
        return

    input_data = normalize_input(input_data)

    # Initialize assistant (OpenAI key from environment if available)
    openai_key = os.getenv("OPENAI_API_KEY")
    assistant = AIAssistant(openai_api_key=openai_key)

    # Perform analysis
    result = assistant.analyze(input_data, explanation=args.explain)
    print("\nAnalysis Result:")
    print(result)

if __name__ == "__main__":
    main()
