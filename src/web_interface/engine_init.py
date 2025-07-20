import streamlit as st
from src.rag_engine import RAGEngine
from src.utils.helpers import create_directories

@st.cache_resource
def initialize_rag_engine():
    """Initialize RAG engine with caching"""
    try:
        create_directories()
        rag_engine = RAGEngine()
        return rag_engine
    except Exception as e:
        st.error(f"Failed to initialize RAG engine: {str(e)}")
        return None
