#!/usr/bin/env python3
"""
Simple launcher for the LangChain RAG web interface
"""

import os
import sys
import subprocess

def main():
    """Launch the Streamlit web interface"""
    # Get the project root directory (parent of 'scripts')
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Add src to Python path
    src_path = os.path.join(project_root, 'src')
    sys.path.insert(0, src_path)
    
    # Path to the Streamlit app
    app_path = os.path.join(src_path, 'web_interface', 'app.py')
    
    print("🚀 Starting LangChain RAG Web Interface...")
    print(f"📁 App path: {app_path}")
    print("🌐 Opening browser to http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()