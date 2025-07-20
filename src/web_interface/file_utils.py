import os
from typing import Optional
from src.utils.helpers import sanitize_filename
import streamlit as st

def save_uploaded_file(uploaded_file) -> Optional[str]:
    """Save uploaded file to documents directory"""
    try:
        os.makedirs("documents", exist_ok=True)
        filename = sanitize_filename(uploaded_file.name)
        file_path = os.path.join("documents", filename)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None
