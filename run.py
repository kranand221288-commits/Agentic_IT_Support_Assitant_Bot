#!/usr/bin/env python3
"""
Entry point to initialize data and run the Streamlit app.
"""
import os
import sys
from pathlib import Path
from db_init import initialize_data

ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"

def main():
    initialize_data(DATA_DIR)
    streamlit_file = ROOT / "streamlit_app.py"
    if not streamlit_file.exists():
        print("streamlit_app.py not found. Exiting.")
        sys.exit(1)
    os.execvp(sys.executable, [sys.executable, "-m", "streamlit", "run", str(streamlit_file)])

if __name__ == "__main__":
    main()
