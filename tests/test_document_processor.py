"""Tests for document processor module"""
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.document_processor.processor import AdvancedDocumentProcessor


class TestAdvancedDocumentProcessor:
    """Test document processing functionality"""
    
    @pytest.fixture
    def processor(self):
        """Create a test document processor"""
        return AdvancedDocumentProcessor(
            chunk_size=500,
            chunk_overlap=50,
            chunking_strategy="recursive"
        )
    
    @pytest.fixture
    def sample_text_file(self):
        """Create a temporary text file for testing"""
        content = """This is a test document.
        
It contains multiple paragraphs with different types of content.

Some technical information:
- Item 1
- Item 2
- Item 3

And some code:
```python
def hello():
    print("Hello World")
```

The end."""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 50
        assert processor.chunking_strategy == "recursive"
        
    def test_content_type_detection(self, processor):
        """Test content type detection"""
        # Test code detection
        code_text = "def function():\n    return True"
        assert processor._detect_content_type(code_text) == "code"
        
        # Test markdown detection
        md_text = "# Header\n## Subheader\n- List item"
        assert processor._detect_content_type(md_text) == "markdown"
        
        # Test plain text (returns semantic for regular text)
        plain_text = "This is just plain text content."
        assert processor._detect_content_type(plain_text) == "semantic"
    
    def test_document_loading(self, processor, sample_text_file):
        """Test document loading from file"""
        docs = processor.load_document(sample_text_file)
        assert len(docs) >= 1
        assert docs[0].page_content is not None
        assert docs[0].metadata['source'] == sample_text_file
        
    def test_document_processing(self, processor, sample_text_file):
        """Test full document processing pipeline"""
        chunks = processor.process_document(sample_text_file)
        
        # Should have chunks
        assert len(chunks) > 0
        
        # Each chunk should have content and metadata
        for chunk in chunks:
            assert chunk.page_content is not None
            assert 'source' in chunk.metadata
            assert 'chunk_position' in chunk.metadata
            assert 'content_type' in chunk.metadata
            
    def test_get_document_info(self, processor, sample_text_file):
        """Test document info extraction"""
        info = processor.get_document_info(sample_text_file)
        
        assert 'file_size' in info
        assert 'content_types' in info
        assert 'total_chunks' in info  # Changed from estimated_chunks
        assert info['file_size'] > 0
        
    def test_chunk_metadata_enrichment(self, processor, sample_text_file):
        """Test that chunks have proper metadata"""
        chunks = processor.process_document(sample_text_file)
        
        for i, chunk in enumerate(chunks):
            assert chunk.metadata['chunk_position'] == f"{i+1}/{len(chunks)}"  # Format is "1/2" not 0
            assert 'content_type' in chunk.metadata
            assert chunk.metadata['source'] == os.path.basename(sample_text_file)
