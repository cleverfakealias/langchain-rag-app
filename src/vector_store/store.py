import os
import sys
from typing import List, Optional, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import FakeEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import chromadb
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add the src directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config.config import Config

logger = logging.getLogger(__name__)

class EnhancedVectorStore:
    """Enhanced vector store with MMR and advanced retrieval methods"""
    
    def __init__(self, persist_directory: str = None):
        # Get configuration
        self.config = Config()
        self.embedding_config = self.config.get_current_embedding_config()
        
        self.persist_directory = persist_directory or self.config.VECTOR_DB_PATH
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_config.name,
                model_kwargs={
                    'device': self.embedding_config.device
                }
            )
        except Exception as e:
            # Fall back to a lightweight in-memory embedding implementation
            logger.warning(
                "Failed to load HuggingFace embeddings '%s'. Falling back to FakeEmbeddings. Error: %s",
                self.embedding_config.name,
                e,
            )
            self.embeddings = FakeEmbeddings(size=384)
            # Reflect the actual embedding model used in stats
            self.embedding_config.name = "fake"
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
    
    def mmr_search(self, query: str, k: int = 4, lambda_mult: float = 0.5, fetch_k: int = 20) -> List[Document]:
        """
        Perform Maximal Marginal Relevance (MMR) search for better diversity
        
        Args:
            query: Search query
            k: Number of documents to return
            lambda_mult: Diversity parameter (0 = max diversity, 1 = max relevance)
            fetch_k: Number of documents to fetch before applying MMR
        """
        try:
            # Get more documents than needed for MMR selection
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=fetch_k)
            
            if not docs_and_scores:
                return []
            
            # Extract documents and scores
            docs = [doc for doc, score in docs_and_scores]
            scores = [score for doc, score in docs_and_scores]
            
            # Apply MMR selection
            selected_docs = self._apply_mmr(docs, scores, query, k, lambda_mult)
            
            logger.info(f"MMR search returned {len(selected_docs)} diverse documents for query: {query[:50]}...")
            return selected_docs
            
        except Exception as e:
            logger.error(f"Error performing MMR search: {str(e)}")
            # Fallback to regular similarity search
            return self.similarity_search(query, k)
    
    def mmr_search_with_score(self, query: str, k: int = 4, lambda_mult: float = 0.5, fetch_k: int = 20) -> List[Tuple[Document, float]]:
        """
        Perform MMR search with scores
        """
        try:
            # Get more documents than needed for MMR selection
            docs_and_scores = self.vector_store.similarity_search_with_score(query, k=fetch_k)
            
            if not docs_and_scores:
                return []
            
            # Extract documents and scores
            docs = [doc for doc, score in docs_and_scores]
            scores = [score for doc, score in docs_and_scores]
            
            # Apply MMR selection
            selected_indices = self._apply_mmr_indices(docs, scores, query, k, lambda_mult)
            
            # Return selected documents with their original scores
            selected_results = [(docs[i], scores[i]) for i in selected_indices]
            
            logger.info(f"MMR search with scores returned {len(selected_results)} diverse documents")
            return selected_results
            
        except Exception as e:
            logger.error(f"Error performing MMR search with scores: {str(e)}")
            # Fallback to regular similarity search with scores
            return self.similarity_search_with_score(query, k)
    
    def _apply_mmr(self, docs: List[Document], scores: List[float], query: str, k: int, lambda_mult: float) -> List[Document]:
        """Apply MMR algorithm to select diverse documents"""
        if len(docs) <= k:
            return docs
        
        # Convert scores to numpy array (lower is better for distance)
        relevance_scores = np.array(scores)
        
        # Get embeddings for documents
        doc_embeddings = self.embeddings.embed_documents([doc.page_content for doc in docs])
        doc_embeddings = np.array(doc_embeddings)
        
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Calculate query-document similarities
        query_doc_similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Initialize selected documents
        selected_indices = []
        remaining_indices = list(range(len(docs)))
        
        # Select first document (most relevant)
        first_idx = np.argmax(query_doc_similarities)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Select remaining documents using MMR
        for _ in range(k - 1):
            if not remaining_indices:
                break
            
            # Calculate MMR scores for remaining documents
            mmr_scores = []
            for idx in remaining_indices:
                # Relevance to query
                relevance = query_doc_similarities[idx]
                
                # Diversity (minimum similarity to already selected documents)
                if selected_indices:
                    selected_embeddings = doc_embeddings[selected_indices]
                    similarities = cosine_similarity(
                        doc_embeddings[idx].reshape(1, -1), 
                        selected_embeddings
                    )[0]
                    diversity = 1 - np.max(similarities)
                else:
                    diversity = 1
                
                # MMR score
                mmr_score = lambda_mult * relevance + (1 - lambda_mult) * diversity
                mmr_scores.append(mmr_score)
            
            # Select document with highest MMR score
            best_idx_in_remaining = np.argmax(mmr_scores)
            best_idx = remaining_indices[best_idx_in_remaining]
            
            selected_indices.append(best_idx)
            remaining_indices.pop(best_idx_in_remaining)
        
        return [docs[i] for i in selected_indices]
    
    def _apply_mmr_indices(self, docs: List[Document], scores: List[float], query: str, k: int, lambda_mult: float) -> List[int]:
        """Apply MMR algorithm and return selected indices"""
        if len(docs) <= k:
            return list(range(len(docs)))
        
        # Convert scores to numpy array
        relevance_scores = np.array(scores)
        
        # Get embeddings for documents
        doc_embeddings = self.embeddings.embed_documents([doc.page_content for doc in docs])
        doc_embeddings = np.array(doc_embeddings)
        
        # Get query embedding
        query_embedding = self.embeddings.embed_query(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        
        # Calculate query-document similarities
        query_doc_similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Initialize selected documents
        selected_indices = []
        remaining_indices = list(range(len(docs)))
        
        # Select first document (most relevant)
        first_idx = np.argmax(query_doc_similarities)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Select remaining documents using MMR
        for _ in range(k - 1):
            if not remaining_indices:
                break
            
            # Calculate MMR scores for remaining documents
            mmr_scores = []
            for idx in remaining_indices:
                # Relevance to query
                relevance = query_doc_similarities[idx]
                
                # Diversity (minimum similarity to already selected documents)
                if selected_indices:
                    selected_embeddings = doc_embeddings[selected_indices]
                    similarities = cosine_similarity(
                        doc_embeddings[idx].reshape(1, -1), 
                        selected_embeddings
                    )[0]
                    diversity = 1 - np.max(similarities)
                else:
                    diversity = 1
                
                # MMR score
                mmr_score = lambda_mult * relevance + (1 - lambda_mult) * diversity
                mmr_scores.append(mmr_score)
            
            # Select document with highest MMR score
            best_idx_in_remaining = np.argmax(mmr_scores)
            best_idx = remaining_indices[best_idx_in_remaining]
            
            selected_indices.append(best_idx)
            remaining_indices.pop(best_idx_in_remaining)
        
        return selected_indices
    
    def hybrid_search(self, query: str, k: int = 4, alpha: float = 0.5) -> List[Document]:
        """
        Perform hybrid search combining semantic and keyword matching
        
        Args:
            query: Search query
            k: Number of documents to return
            alpha: Weight for semantic vs keyword search (0 = keyword only, 1 = semantic only)
        """
        try:
            # Semantic search
            semantic_results = self.similarity_search_with_score(query, k=k*2)
            
            # Simple keyword search (fallback for now)
            # In a more advanced implementation, you could use BM25 or similar
            keyword_results = self._keyword_search(query, k*2)
            
            # Combine results
            combined_results = self._combine_search_results(
                semantic_results, keyword_results, alpha, k
            )
            
            logger.info(f"Hybrid search returned {len(combined_results)} documents")
            return combined_results
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {str(e)}")
            return self.similarity_search(query, k)
    
    def _keyword_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Simple keyword-based search"""
        try:
            # Get all documents
            collection = self.vector_store._collection
            results = collection.get()
            
            if not results or not results['documents']:
                return []
            
            # Simple keyword matching
            query_words = set(query.lower().split())
            scored_docs = []
            
            for i, doc_content in enumerate(results['documents']):
                doc_words = set(doc_content.lower().split())
                intersection = query_words.intersection(doc_words)
                
                if intersection:
                    # Simple scoring based on word overlap
                    score = len(intersection) / len(query_words)
                    
                    # Create document object
                    doc = Document(
                        page_content=doc_content,
                        metadata=results['metadatas'][i] if results['metadatas'] else {}
                    )
                    
                    scored_docs.append((doc, score))
            
            # Sort by score and return top k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return scored_docs[:k]
            
        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            return []
    
    def _combine_search_results(self, semantic_results: List[Tuple[Document, float]], 
                               keyword_results: List[Tuple[Document, float]], 
                               alpha: float, k: int) -> List[Document]:
        """Combine semantic and keyword search results"""
        # Create a dictionary to store combined scores
        doc_scores = {}
        
        # Add semantic scores
        for doc, score in semantic_results:
            doc_id = doc.page_content[:100]  # Simple ID based on content
            doc_scores[doc_id] = {
                'doc': doc,
                'semantic_score': score,
                'keyword_score': 0
            }
        
        # Add keyword scores
        for doc, score in keyword_results:
            doc_id = doc.page_content[:100]
            if doc_id in doc_scores:
                doc_scores[doc_id]['keyword_score'] = score
            else:
                doc_scores[doc_id] = {
                    'doc': doc,
                    'semantic_score': 0,
                    'keyword_score': score
                }
        
        # Calculate combined scores
        combined_results = []
        for doc_id, scores in doc_scores.items():
            combined_score = alpha * scores['semantic_score'] + (1 - alpha) * scores['keyword_score']
            combined_results.append((scores['doc'], combined_score))
        
        # Sort by combined score and return top k
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in combined_results[:k]]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                "total_documents": count,
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_config.name
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
                # Use where parameter instead of filter for ChromaDB
                self.vector_store.delete(where=filter_dict)
            else:
                # Delete all documents by getting all IDs first
                collection = self.vector_store._collection
                results = collection.get()
                if results and results['ids']:
                    self.vector_store.delete(ids=results['ids'])
                else:
                    logger.info("No documents to delete")
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

# Keep the original class for backward compatibility
class VectorStore(EnhancedVectorStore):
    """Backward compatibility wrapper"""
    pass 