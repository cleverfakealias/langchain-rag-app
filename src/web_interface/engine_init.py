import streamlit as st
from src.rag_engine import EnhancedRAGEngine
from src.utils.helpers import create_directories

@st.cache_resource
def initialize_rag_engine():
    """Initialize enhanced RAG engine with caching"""
    try:
        create_directories()
        rag_engine = EnhancedRAGEngine()
        return rag_engine
    except Exception as e:
        st.error(f"Failed to initialize RAG engine: {str(e)}")
        return None
