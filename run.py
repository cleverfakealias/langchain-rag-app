#!/usr/bin/env python3
"""
Simple launcher for the LangChain RAG Application
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application"""
    try:
        # Check if streamlit is installed
        subprocess.run([sys.executable, "-m", "streamlit", "--version"], 
                      check=True, capture_output=True)
        
        # Run the Streamlit app
        app_path = os.path.join("src", "web_interface", "app.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
        
    except subprocess.CalledProcessError:
        print("❌ Streamlit is not installed. Please run: pip install streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 