"""Tests for vector store module"""
import pytest
import tempfile
import shutil
import os
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.vector_store.store import EnhancedVectorStore


class TestEnhancedVectorStore:
    """Test vector store functionality"""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary directory for test database"""
        import uuid
        temp_dir = tempfile.mkdtemp(prefix=f"test_vector_db_{uuid.uuid4().hex[:8]}_")
        yield temp_dir
        # Cleanup with retry for Windows file locking
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                break
            except (PermissionError, OSError) as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Wait before retry
                    continue
                # On final attempt, just log the warning
                import warnings
                warnings.warn(f"Could not clean up temp directory {temp_dir}: {e}")
    
    @pytest.fixture
    def vector_store(self, temp_db_path):
        """Create a vector store with proper cleanup"""
        store = EnhancedVectorStore(persist_directory=temp_db_path)
        yield store
        # Cleanup
        try:
            if hasattr(store, 'vector_store') and store.vector_store:
                store.vector_store._client.reset()
        except:
            pass
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                page_content="This is about artificial intelligence and machine learning.",
                metadata={"source": "doc1.txt", "content_type": "text"}
            ),
            Document(
                page_content="Python is a programming language used for data science.",
                metadata={"source": "doc2.txt", "content_type": "code"}
            ),
            Document(
                page_content="Vector databases store embeddings for similarity search.",
                metadata={"source": "doc3.txt", "content_type": "text"}
            )
        ]
    
    def test_vector_store_initialization(self, temp_db_path):
        """Test vector store initialization"""
        store = EnhancedVectorStore(persist_directory=temp_db_path)
        try:
            assert store.persist_directory == temp_db_path
            assert store.embeddings is not None
        finally:
            # Ensure proper cleanup
            if hasattr(store, 'vector_store') and store.vector_store:
                try:
                    store.vector_store._client.reset()
                except:
                    pass
        
    def test_add_documents(self, vector_store, sample_documents):
        """Test adding documents to vector store"""
        # Add documents
        vector_store.add_documents(sample_documents)
        
        # Verify they were added
        stats = vector_store.get_collection_stats()
        assert stats['total_documents'] >= len(sample_documents)
        
    def test_similarity_search(self, temp_db_path, sample_documents):
        """Test similarity search functionality"""
        store = EnhancedVectorStore(persist_directory=temp_db_path)
        store.add_documents(sample_documents)
        
        # Search for relevant documents
        results = store.similarity_search("machine learning AI", k=2)
        
        assert len(results) > 0
        assert all(isinstance(doc, Document) for doc in results)
        
    def test_similarity_search_with_score(self, temp_db_path, sample_documents):
        """Test similarity search with scores"""
        store = EnhancedVectorStore(persist_directory=temp_db_path)
        store.add_documents(sample_documents)
        
        # Search with scores
        results = store.similarity_search_with_score("Python programming", k=2)
        
        assert len(results) > 0
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)
            
    def test_mmr_search(self, temp_db_path, sample_documents):
        """Test MMR (Maximum Marginal Relevance) search"""
        store = EnhancedVectorStore(persist_directory=temp_db_path)
        store.add_documents(sample_documents)
        
        # MMR search for diversity
        results = store.mmr_search("programming data science", k=2, lambda_mult=0.7)
        
        assert len(results) > 0
        assert all(isinstance(doc, Document) for doc in results)
        
    def test_get_document_sources(self, temp_db_path, sample_documents):
        """Test getting document sources"""
        store = EnhancedVectorStore(persist_directory=temp_db_path)
        store.add_documents(sample_documents)
        
        sources = store.get_document_sources()
        expected_sources = ["doc1.txt", "doc2.txt", "doc3.txt"]
        
        for source in expected_sources:
            assert source in sources
            
    def test_collection_stats(self, temp_db_path, sample_documents):
        """Test collection statistics"""
        store = EnhancedVectorStore(persist_directory=temp_db_path)
        store.add_documents(sample_documents)
        
        stats = store.get_collection_stats()
        
        assert 'total_documents' in stats
        assert 'embedding_model' in stats
        assert stats['total_documents'] >= len(sample_documents)
        
    def test_delete_documents(self, temp_db_path, sample_documents):
        """Test document deletion"""
        store = EnhancedVectorStore(persist_directory=temp_db_path)
        store.add_documents(sample_documents)
        
        # Delete documents from specific source
        store.delete_documents({"source": "doc1.txt"})
        
        # Verify deletion
        sources = store.get_document_sources()
        assert "doc1.txt" not in sources
        
    def test_clear_all(self, temp_db_path, sample_documents):
        """Test clearing all documents"""
        store = EnhancedVectorStore(persist_directory=temp_db_path)
        store.add_documents(sample_documents)
        
        # Clear all documents
        store.clear_all()
        
        # Verify clearing
        stats = store.get_collection_stats()
        assert stats['total_documents'] == 0
