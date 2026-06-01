import os

def setup_environment():
    """Create necessary folders if they don't exist."""
    for folder in ["uploads", "results"]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"📂 Created folder: {folder}")

def main():
    print("🔍 Starting Fingerprint Forensics Toolkit...")

    # Step 1: Setup environment
    setup_environment()

    # Step 2: Launch Streamlit UI (Person A orchestrator)
    os.system("streamlit run \"Fingerprint Main/app.py\"")

if __name__ == "__main__":
    main()
