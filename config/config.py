"""
Configuration settings for the LangChain RAG Application
"""

import os
from typing import Dict, Any

class Config:
    """Application configuration"""
    
    # OpenAI Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    
    # Document Processing Settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Vector Store Settings
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vector_db")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # Retrieval Settings
    NUM_RETRIEVED_DOCS = int(os.getenv("NUM_RETRIEVED_DOCS", "4"))
    
    # File Storage Settings
    DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "documents")
    
    # Supported file types
    SUPPORTED_FILE_TYPES = [".pdf", ".txt", ".md", ".docx"]
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        return True
    
    @classmethod
    def get_settings(cls) -> Dict[str, Any]:
        """Get all settings as a dictionary"""
        return {
            "model_name": cls.MODEL_NAME,
            "temperature": cls.TEMPERATURE,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "vector_db_path": cls.VECTOR_DB_PATH,
            "embedding_model": cls.EMBEDDING_MODEL,
            "num_retrieved_docs": cls.NUM_RETRIEVED_DOCS,
            "documents_path": cls.DOCUMENTS_PATH,
            "supported_file_types": cls.SUPPORTED_FILE_TYPES
        } 