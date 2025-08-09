import os
from typing import List, Dict, Any, Optional
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    TokenTextSplitter
)
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader
)
from langchain_core.documents import Document
import logging
import re

logger = logging.getLogger(__name__)

class AdvancedDocumentProcessor:
    """Enhanced document processor with advanced chunking strategies"""
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 chunking_strategy: str = "recursive"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        
        # Initialize different text splitters
        self._initialize_text_splitters()
    
    def _initialize_text_splitters(self):
        """Initialize different text splitters for various strategies"""
        
        # Recursive character splitter (default)
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        
        # Token-based splitter (for more precise control)
        try:
            self.token_splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                encoding_name="cl100k_base"  # GPT-4 tokenizer
            )
        except Exception as e:
            # Avoid hard failure when tokenizer files aren't available
            logger.warning(
                "TokenTextSplitter failed to load 'cl100k_base' tokenizer. "
                "Falling back to a simple character splitter. Error: %s", e
            )
            self.token_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        
        # Markdown-aware splitter
        self.markdown_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ",  # Headers
                "\n\n", "\n", ". ", "! ", "? ", " ", ""  # Regular separators
            ]
        )
        
        # Code-aware splitter
        self.code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n```", "\nclass ", "\ndef ", "\nif ", "\nfor ", "\nwhile ",  # Code blocks
                "\n\n", "\n", ". ", "! ", "? ", " ", ""  # Regular separators
            ]
        )
        
        # Semantic splitter (tries to keep semantic units together)
        self.semantic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""  # Prioritize paragraphs
            ]
        )
    
    def _detect_content_type(self, text: str) -> str:
        """Detect the type of content to choose appropriate chunking strategy"""
        text_lower = text.lower()
        
        # Check for code content
        if any(keyword in text_lower for keyword in [
            "def ", "class ", "import ", "from ", "if __name__", 
            "function", "var ", "const ", "let ", "public ", "private ",
            "```", "<?php", "<script", "<html", "package ", "namespace "
        ]):
            return "code"
        
        # Check for markdown content
        if any(pattern in text for pattern in [
            "# ", "## ", "### ", "#### ", "##### ", "###### ",
            "**", "*", "`", "```", "[", "]", "!", ">"
        ]):
            return "markdown"
        
        # Check for structured content (lists, tables, etc.)
        if any(pattern in text for pattern in [
            "\n- ", "\n* ", "\n1. ", "\n2. ", "\n3. ",
            "\n|", "\n---", "\n=", "\n=="
        ]):
            return "structured"
        
        # Default to semantic chunking for regular text
        return "semantic"
    
    def _choose_splitter(self, content_type: str) -> RecursiveCharacterTextSplitter:
        """Choose the appropriate text splitter based on content type"""
        if content_type == "code":
            return self.code_splitter
        elif content_type == "markdown":
            return self.markdown_splitter
        elif content_type == "structured":
            return self.semantic_splitter
        else:
            return self.semantic_splitter
    
    def _post_process_chunks(self, chunks: List[Document]) -> List[Document]:
        """Post-process chunks to improve quality"""
        processed_chunks = []
        
        for chunk in chunks:
            # Clean up whitespace
            cleaned_content = re.sub(r'\n\s*\n\s*\n', '\n\n', chunk.page_content)
            cleaned_content = cleaned_content.strip()
            
            # Skip chunks that are too short (likely incomplete)
            if len(cleaned_content) < 50:
                continue
            
            # Skip chunks that are mostly whitespace or special characters
            if len(re.sub(r'\s', '', cleaned_content)) < 20:
                continue
            
            # Create new document with cleaned content
            new_chunk = Document(
                page_content=cleaned_content,
                metadata=chunk.metadata.copy()
            )
            processed_chunks.append(new_chunk)
        
        return processed_chunks
    
    def _add_chunk_metadata(self, chunks: List[Document], file_path: str) -> List[Document]:
        """Add enhanced metadata to chunks"""
        filename = os.path.basename(file_path)
        file_extension = os.path.splitext(file_path)[1].lower()
        
        for i, chunk in enumerate(chunks):
            # Basic metadata
            chunk.metadata.update({
                "source": filename,
                "file_path": file_path,
                "file_type": file_extension,
                "chunk_id": i,
                "chunk_size": len(chunk.page_content),
                "total_chunks": len(chunks)
            })
            
            # Add content type detection
            content_type = self._detect_content_type(chunk.page_content)
            chunk.metadata["content_type"] = content_type
            
            # Add chunk position info
            chunk.metadata["chunk_position"] = f"{i+1}/{len(chunks)}"
            
            # Add first few words as title
            words = chunk.page_content.split()[:5]
            chunk.metadata["title"] = " ".join(words) + ("..." if len(words) == 5 else "")
        
        return chunks
    
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
        """Split documents into chunks using advanced strategies"""
        try:
            all_chunks = []
            
            for doc in documents:
                # Detect content type for this document
                content_type = self._detect_content_type(doc.page_content)
                
                # Choose appropriate splitter
                splitter = self._choose_splitter(content_type)
                
                # Split the document
                doc_chunks = splitter.split_documents([doc])
                
                # Post-process chunks
                processed_chunks = self._post_process_chunks(doc_chunks)
                
                all_chunks.extend(processed_chunks)
                
                logger.info(f"Split document into {len(processed_chunks)} chunks using {content_type} strategy")
            
            logger.info(f"Total chunks created: {len(all_chunks)}")
            return all_chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise
    
    def process_document(self, file_path: str) -> List[Document]:
        """Complete document processing pipeline with enhanced chunking"""
        # Load document
        documents = self.load_document(file_path)
        
        # Split into chunks using advanced strategies
        split_docs = self.split_documents(documents)
        
        # Add enhanced metadata
        split_docs = self._add_chunk_metadata(split_docs, file_path)
        
        return split_docs
    
    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """Get detailed information about a document"""
        try:
            documents = self.load_document(file_path)
            split_docs = self.split_documents(documents)
            
            # Analyze content types
            content_types = {}
            for doc in split_docs:
                content_type = self._detect_content_type(doc.page_content)
                content_types[content_type] = content_types.get(content_type, 0) + 1
            
            return {
                "filename": os.path.basename(file_path),
                "file_size": os.path.getsize(file_path),
                "total_pages": len(documents),
                "total_chunks": len(split_docs),
                "file_type": os.path.splitext(file_path)[1].lower(),
                "content_types": content_types,
                "chunking_strategy": self.chunking_strategy,
                "avg_chunk_size": sum(len(doc.page_content) for doc in split_docs) / len(split_docs) if split_docs else 0
            }
        except Exception as e:
            logger.error(f"Error getting document info for {file_path}: {str(e)}")
            return {
                "filename": os.path.basename(file_path),
                "error": str(e)
            }

# Keep the original class for backward compatibility
class DocumentProcessor(AdvancedDocumentProcessor):
    """Backward compatibility wrapper"""
    pass 