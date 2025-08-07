import logging
import os
from typing import List, Dict, Any, Optional
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pipelines import pipeline

# Try importing BitsAndBytesConfig for quantization support
try:
    from transformers.utils.quantization_config import BitsAndBytesConfig
    _HAS_BNB = True
except ImportError:
    BitsAndBytesConfig = None
    _HAS_BNB = False

from config.config import Config
from src.document_processor.processor import AdvancedDocumentProcessor
from src.vector_store.store import EnhancedVectorStore

logger = logging.getLogger(__name__)

class EnhancedRAGEngine:
    """Enhanced RAG engine with MMR and advanced retrieval methods"""
    
    def __init__(self, model_name: Optional[str] = None):
        # Get configuration
        self.config = Config()
        self.llm_config = self.config.get_current_llm_config()
        self.model_name = model_name or self.llm_config.name
        
        # Initialize components with enhanced capabilities
        self.document_processor = AdvancedDocumentProcessor(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            chunking_strategy=self.config.CHUNKING_STRATEGY
        )
        self.vector_store = EnhancedVectorStore()
        self.llm = self._initialize_huggingface_model()
        
        # Initialize conversation chain
        self.conversation_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Retrieval configuration
        self.retrieval_config = {
            "method": self.config.RETRIEVAL_METHOD,  # Options: "similarity", "mmr", "hybrid"
            "k": self.config.NUM_RETRIEVED_DOCS,
            "mmr_lambda": self.config.MMR_LAMBDA,  # MMR diversity parameter
            "fetch_k": self.config.FETCH_K,  # Number of docs to fetch before MMR selection
            "hybrid_alpha": self.config.HYBRID_ALPHA  # Weight for semantic vs keyword search
        }
        
        self._setup_conversation_chain()
    
    def _initialize_huggingface_model(self):
        """Initialize HuggingFace model with multiple fallback configurations"""
        # Get all available configurations
        configs_to_try = self.config.get_all_llm_configs()
        
        for i, config in enumerate(configs_to_try):
            try:
                logger.info(f"Attempting to load model with config {i+1}: {config['name']}")
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    config['name'],
                    cache_dir=self.config.CACHE_DIR,
                    trust_remote_code=True
                )
                
                # Set padding token if not present
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model with quantization if available
                model_kwargs = {
                    "cache_dir": self.config.CACHE_DIR,
                    "trust_remote_code": True,
                    "device_map": config['device_map'],
                    "torch_dtype": config['torch_dtype']
                }
                
                if config['load_in_8bit'] and _HAS_BNB:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0,
                        llm_int8_has_fp16_weight=False
                    )
                
                model = AutoModelForCausalLM.from_pretrained(
                    config['name'],
                    **model_kwargs
                )

                # Create pipeline with optimized configuration
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=self.llm_config.max_new_tokens,
                    temperature=config['temperature'],
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2,
                    top_p=0.95,
                    top_k=40,
                    return_full_text=False,
                    eos_token_id=tokenizer.eos_token_id
                )

                # Create LangChain wrapper
                llm = HuggingFacePipeline(pipeline=pipe)

                logger.info(f"HuggingFace model loaded successfully with config {i+1}")
                return llm

            except Exception as e:
                logger.warning(f"Failed to load model with config {i+1}: {str(e)}")
                if i == len(configs_to_try) - 1:
                    logger.error(f"All model loading attempts failed. Last error: {str(e)}")
                    raise
                continue
    
    def _setup_conversation_chain(self):
        """Setup the conversational retrieval chain with enhanced retrieval"""
        try:
            # Custom prompt template optimized for better responses
            prompt_template = """Using the following context, answer the question in a friendly and detailed manner. 

- If you can answer directly, provide a clear paragraph first.
- If there are multiple key points, include a summary as bullet points after the paragraph.
- If you are not certain, say so, and provide your best summary based on the context.
- Cite the source document(s) in your answer. If there are multiple sources, list them at the end.
- If the context contains code, format it properly and explain what it does.

Context:
{context}

Question:
{question}

Answer:
"""

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            # Check if vector_store.vector_store is initialized
            if not hasattr(self.vector_store, "vector_store") or self.vector_store.vector_store is None:
                raise ValueError("Vector store is not initialized properly.")

            # Create conversation chain with enhanced settings
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.vector_store.as_retriever(
                    search_kwargs={"k": self.retrieval_config["fetch_k"]}
                ),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                return_source_documents=True,
                verbose=False,
                rephrase_question=False,
                max_tokens_limit=2000
            )

            logger.info("Enhanced conversation chain initialized successfully")

        except Exception as e:
            logger.error(f"Error setting up conversation chain: {str(e)}")
            raise
    
    def set_retrieval_config(self, method: str = "similarity", **kwargs):
        """Set retrieval configuration"""
        self.vector_store.set_retrieval_config(method=method, **kwargs)
        logger.info(f"Updated retrieval config: {method}")
    
    def set_chunking_config(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Set chunking configuration"""
        self.document_processor.set_chunking_config(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logger.info(f"Updated chunking config: size={chunk_size}, overlap={chunk_overlap}")
    
    def _retrieve_documents(self, query: str) -> List[Document]:
        """Retrieve documents using the configured retrieval method"""
        method = self.retrieval_config["method"]
        k = self.retrieval_config["k"]
        
        try:
            if method == "mmr":
                lambda_mult = self.retrieval_config["mmr_lambda"]
                fetch_k = self.retrieval_config["fetch_k"]
                return self.vector_store.mmr_search(query, k=k, lambda_mult=lambda_mult, fetch_k=fetch_k)
            
            elif method == "hybrid":
                alpha = self.retrieval_config["hybrid_alpha"]
                return self.vector_store.hybrid_search(query, k=k, alpha=alpha)
            
            else:  # Default to similarity search
                return self.vector_store.similarity_search(query, k=k)
                
        except Exception as e:
            logger.error(f"Error in {method} retrieval: {str(e)}")
            # Fallback to similarity search
            return self.vector_store.similarity_search(query, k=k)
    
    def process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process multiple documents and add them to the vector store"""
        results = {
            "success": [],
            "errors": [],
            "total_processed": 0,
            "total_chunks": 0,
            "content_analysis": {}
        }
        
        for file_path in file_paths:
            try:
                # Process document with enhanced processor
                documents = self.document_processor.process_document(file_path)
                
                # Add to vector store
                self.vector_store.add_documents(documents)
                
                # Get detailed document info
                doc_info = self.document_processor.get_document_info(file_path)
                
                # Update results
                results["success"].append({
                    "file_path": file_path,
                    "filename": os.path.basename(file_path),
                    "chunks": len(documents),
                    "content_types": doc_info.get("content_types", {}),
                    "avg_chunk_size": doc_info.get("avg_chunk_size", 0)
                })
                results["total_processed"] += 1
                results["total_chunks"] += len(documents)
                
                # Aggregate content analysis
                for content_type, count in doc_info.get("content_types", {}).items():
                    results["content_analysis"][content_type] = results["content_analysis"].get(content_type, 0) + count
                
                logger.info(f"Successfully processed {file_path} with {len(documents)} chunks")
                
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                results["errors"].append({
                    "file_path": file_path,
                    "error": str(e)
                })
                logger.error(error_msg)
        
        return results
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an answer with enhanced retrieval"""
        try:
            if not self.conversation_chain:
                raise ValueError("Conversation chain not initialized")
            
            # Get response with error handling
            try:
                response = self.conversation_chain.invoke({"question": question})
                
                # Check if response is empty or None
                answer = response.get("answer", "")
                if not answer or answer.strip() == "":
                    logger.warning("LLM returned empty response, falling back to enhanced document search")
                    raise ValueError("Empty response from LLM")
                
                # Check if response is too short or nonsensical
                if len(answer.strip()) < 10 or "I don't" in answer.lower() or "cannot" in answer.lower():
                    logger.warning("LLM returned very short or unclear response, falling back to enhanced document search")
                    raise ValueError("Unclear response from LLM")
                    
            except Exception as generation_error:
                logger.error(f"Generation error: {str(generation_error)}")
                # Try enhanced retrieval approach
                docs = self._retrieve_documents(question)
                if docs:
                    # Create a more structured answer from retrieved documents
                    answer = f"Based on your documents, here's what I found:\n\n"
                    
                    # Group by source for better organization
                    docs_by_source = {}
                    for doc in docs:
                        source = doc.metadata.get("source", "Unknown")
                        if source not in docs_by_source:
                            docs_by_source[source] = []
                        docs_by_source[source].append(doc)
                    
                    for source, source_docs in docs_by_source.items():
                        answer += f"**From {source}:**\n"
                        for i, doc in enumerate(source_docs, 1):
                            content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                            answer += f"{i}. {content}\n\n"
                    
                    # Add retrieval method info
                    method = self.retrieval_config["method"]
                    answer += f"💡 *Note: This information was retrieved using {method.upper()} search from your documents.*"
                    
                    sources = [{"source": doc.metadata.get("source", "Unknown"), 
                               "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                               "content_type": doc.metadata.get("content_type", "unknown"),
                               "chunk_position": doc.metadata.get("chunk_position", "unknown")} 
                              for doc in docs]
                    return {
                        "answer": answer,
                        "sources": sources,
                        "question": question,
                        "retrieval_method": method
                    }
                else:
                    return {
                        "answer": "I couldn't find any relevant information in the documents to answer your question. Please try rephrasing your question or check if the documents have been processed correctly.",
                        "sources": [],
                        "question": question,
                        "retrieval_method": self.retrieval_config["method"]
                    }
            
            # Extract sources with enhanced metadata
            sources = []
            if response.get("source_documents"):
                for doc in response["source_documents"]:
                    if doc.metadata.get("source"):
                        sources.append({
                            "source": doc.metadata["source"],
                            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                            "content_type": doc.metadata.get("content_type", "unknown"),
                            "chunk_position": doc.metadata.get("chunk_position", "unknown")
                        })
            
            return {
                "answer": answer,
                "sources": sources,
                "question": question,
                "retrieval_method": self.retrieval_config["method"]
            }
            
        except Exception as e:
            logger.error(f"Error asking question: {str(e)}")
            return {
                "answer": f"Sorry, I encountered an error while processing your question. Please try rephrasing it or ask a different question.",
                "sources": [],
                "question": question,
                "retrieval_method": self.retrieval_config["method"]
            }
    
    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """Get detailed information about a document without processing it"""
        return self.document_processor.get_document_info(file_path)
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return self.vector_store.get_collection_stats()
    
    def get_document_sources(self) -> List[str]:
        """Get list of document sources in the vector store"""
        return self.vector_store.get_document_sources()
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """Get comprehensive document statistics"""
        try:
            # Get vector store stats
            vector_stats = self.vector_store.get_collection_stats()
            vector_sources = self.vector_store.get_document_sources()
            
            # Get physical document stats
            documents_dir = "documents"
            physical_docs = []
            total_physical_size = 0
            
            if os.path.exists(documents_dir):
                for file in os.listdir(documents_dir):
                    if file.lower().endswith(('.pdf', '.txt', '.md', '.docx')):
                        file_path = os.path.join(documents_dir, file)
                        file_size = os.path.getsize(file_path)
                        physical_docs.append({
                            'name': file,
                            'size': file_size,
                            'indexed': file in vector_sources
                        })
                        total_physical_size += file_size
            
            return {
                'vector_store': {
                    'total_documents': vector_stats.get('total_documents', 0),
                    'sources': vector_sources,
                    'embedding_model': vector_stats.get('embedding_model', 'Unknown')
                },
                'physical_files': {
                    'total_files': len(physical_docs),
                    'total_size': total_physical_size,
                    'files': physical_docs
                },
                'summary': {
                    'indexed_files': len([doc for doc in physical_docs if doc['indexed']]),
                    'unindexed_files': len([doc for doc in physical_docs if not doc['indexed']]),
                    'orphaned_vectors': len([s for s in vector_sources if s not in [doc['name'] for doc in physical_docs]])
                }
            }
        except Exception as e:
            logger.error(f"Error getting document statistics: {str(e)}")
            return {
                'error': str(e)
            }
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def clear_vector_store(self):
        """Clear all documents from vector store"""
        self.vector_store.clear_all()
        logger.info("Vector store cleared")
    
    def delete_document_from_vector_store(self, filename: str):
        """Delete a specific document from vector store by filename"""
        try:
            # Delete documents with matching source filename
            self.vector_store.delete_documents({"source": filename})
            logger.info(f"Deleted document {filename} from vector store")
        except Exception as e:
            logger.error(f"Error deleting document {filename} from vector store: {str(e)}")
            raise
    
    def search_documents(self, query: str, k: int = 4) -> List[Document]:
        """Search documents without generating an answer"""
        return self._retrieve_documents(query)
    
    def search_documents_with_scores(self, query: str, k: int = 4) -> List[tuple]:
        """Search documents with similarity scores using configured method"""
        method = self.retrieval_config["method"]
        
        try:
            if method == "mmr":
                lambda_mult = self.retrieval_config["mmr_lambda"]
                fetch_k = self.retrieval_config["fetch_k"]
                return self.vector_store.mmr_search_with_score(query, k=k, lambda_mult=lambda_mult, fetch_k=fetch_k)
            else:
                return self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as e:
            logger.error(f"Error in search with scores: {str(e)}")
            return self.vector_store.similarity_search_with_score(query, k=k)

# Keep the original class for backward compatibility
class RAGEngine(EnhancedRAGEngine):
    """Backward compatibility wrapper"""
    pass