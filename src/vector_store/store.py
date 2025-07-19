import os
from typing import List, Optional, Dict, Any
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import chromadb
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages vector database operations using ChromaDB"""
    
    def __init__(self, persist_directory: str = "vector_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize or load existing vector store"""
        try:
            if os.path.exists(self.persist_directory):
                # Load existing vector store
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info(f"Loaded existing vector store from {self.persist_directory}")
            else:
                # Create new vector store
                self.vector_store = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info(f"Created new vector store at {self.persist_directory}")
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        try:
            if not documents:
                logger.warning("No documents to add")
                return
            
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            
            # Persist changes
            self.vector_store.persist()
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search"""
        try:
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} similar documents for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Perform similarity search with scores"""
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} similar documents with scores for query: {query[:50]}...")
            return results
        except Exception as e:
            logger.error(f"Error performing similarity search with scores: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                "total_documents": count,
                "persist_directory": self.persist_directory,
                "embedding_model": "text-embedding-ada-002"
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                "error": str(e)
            }
    
    def delete_documents(self, filter_dict: Optional[Dict] = None) -> None:
        """Delete documents from vector store"""
        try:
            if filter_dict:
                self.vector_store.delete(filter=filter_dict)
            else:
                # Delete all documents
                self.vector_store.delete()
            
            self.vector_store.persist()
            logger.info("Documents deleted from vector store")
            
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            raise
    
    def clear_all(self) -> None:
        """Clear all documents from vector store"""
        try:
            self.delete_documents()
            logger.info("Vector store cleared")
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            raise
    
    def get_document_sources(self) -> List[str]:
        """Get list of unique document sources"""
        try:
            collection = self.vector_store._collection
            results = collection.get()
            
            if results and results['metadatas']:
                sources = set()
                for metadata in results['metadatas']:
                    if metadata and 'source' in metadata:
                        sources.add(metadata['source'])
                return list(sources)
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting document sources: {str(e)}")
            return [] 