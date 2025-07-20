#!/usr/bin/env python3
"""
Main entry point for the LangChain RAG Application
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    # Import and run the Streamlit app
    from src.web_interface.app import main
    main() 