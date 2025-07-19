import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader
)
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load document based on file type"""
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            elif file_extension == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == ".md":
                loader = UnstructuredMarkdownLoader(file_path)
            elif file_extension == ".docx":
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages/sections from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
            return split_docs
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise
    
    def process_document(self, file_path: str) -> List[Document]:
        """Complete document processing pipeline"""
        # Load document
        documents = self.load_document(file_path)
        
        # Split into chunks
        split_docs = self.split_documents(documents)
        
        # Add metadata
        for doc in split_docs:
            doc.metadata["source"] = os.path.basename(file_path)
            doc.metadata["file_path"] = file_path
        
        return split_docs
    
    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic information about a document"""
        try:
            documents = self.load_document(file_path)
            total_chunks = len(self.split_documents(documents))
            
            return {
                "filename": os.path.basename(file_path),
                "file_size": os.path.getsize(file_path),
                "total_pages": len(documents),
                "total_chunks": total_chunks,
                "file_type": os.path.splitext(file_path)[1].lower()
            }
        except Exception as e:
            logger.error(f"Error getting document info for {file_path}: {str(e)}")
            return {
                "filename": os.path.basename(file_path),
                "error": str(e)
            } 