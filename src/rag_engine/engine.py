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
from src.document_processor.processor import DocumentProcessor
from src.vector_store.store import VectorStore

logger = logging.getLogger(__name__)

class RAGEngine:
    """Main RAG engine that orchestrates document processing, retrieval, and generation"""
    
    def __init__(self, model_name: Optional[str] = None):
        # Get configuration
        self.config = Config()
        self.llm_config = self.config.get_current_llm_config()
        self.model_name = model_name or self.llm_config.name
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm = self._initialize_huggingface_model()
        
        # Initialize conversation chain
        self.conversation_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        self._setup_conversation_chain()
    
    def _initialize_huggingface_model(self):
        """Initialize HuggingFace model for text generation"""
        configs_to_try = [
            {"use_quantization": True, "temperature": self.llm_config.temperature},
            {"use_quantization": False, "temperature": self.llm_config.temperature},
            {"use_quantization": False, "temperature": 0.5},
        ]

        for i, config in enumerate(configs_to_try):
            try:
                logger.info(f"Attempting to load model with config {i+1}: {config}")

                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)

                # Configure quantization if needed and available
                quantization_config = None
                if config["use_quantization"] and self.llm_config.load_in_8bit:
                    if _HAS_BNB and BitsAndBytesConfig is not None:
                        try:
                            quantization_config = BitsAndBytesConfig(
                                load_in_8bit=True,
                                llm_int8_threshold=6.0
                            )
                        except Exception as e:
                            logger.warning(f"Failed to configure 8-bit quantization: {e}. Using full precision.")
                            quantization_config = None
                    else:
                        logger.warning("bitsandbytes is not installed or BitsAndBytesConfig unavailable. Skipping quantization.")

                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=self.llm_config.torch_dtype,
                    device_map=self.llm_config.device_map,
                    quantization_config=quantization_config
                )

                # Create pipeline with GPT-2 optimized configuration
                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=self.llm_config.max_new_tokens,
                    temperature=config["temperature"],
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
        """Setup the conversational retrieval chain"""
        try:
            # Custom prompt template optimized for GPT-2
            prompt_template = """Using the following context, answer the question in a friendly and detailed manner. 

- If you can answer directly, provide a clear paragraph first.
- If there are multiple key points, include a summary as bullet points after the paragraph.
- If you are not certain, say so, and provide your best summary based on the context.
- Cite the source document(s) in your answer. If there are multiple sources, list them at the end.

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

            # Create conversation chain with GPT-2 optimized settings
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.vector_store.as_retriever(
                    search_kwargs={"k": 3}
                ),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                return_source_documents=True,
                verbose=False,
                rephrase_question=False,
                max_tokens_limit=2000
            )

            logger.info("Conversation chain initialized successfully")

        except Exception as e:
            logger.error(f"Error setting up conversation chain: {str(e)}")
            raise
    
    def process_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process multiple documents and add them to the vector store"""
        results = {
            "success": [],
            "errors": [],
            "total_processed": 0,
            "total_chunks": 0
        }
        
        for file_path in file_paths:
            try:
                # Process document
                documents = self.document_processor.process_document(file_path)
                
                # Add to vector store
                self.vector_store.add_documents(documents)
                
                # Update results
                results["success"].append({
                    "file_path": file_path,
                    "filename": os.path.basename(file_path),
                    "chunks": len(documents)
                })
                results["total_processed"] += 1
                results["total_chunks"] += len(documents)
                
                logger.info(f"Successfully processed {file_path}")
                
            except Exception as e:
                error_msg = f"Error processing {file_path}: {str(e)}"
                results["errors"].append({
                    "file_path": file_path,
                    "error": str(e)
                })
                logger.error(error_msg)
        
        return results
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get an answer with sources"""
        try:
            if not self.conversation_chain:
                raise ValueError("Conversation chain not initialized")
            
            # Get response with error handling
            try:
                response = self.conversation_chain.invoke({"question": question})
                
                # Check if response is empty or None
                answer = response.get("answer", "")
                if not answer or answer.strip() == "":
                    logger.warning("LLM returned empty response, falling back to document search")
                    raise ValueError("Empty response from LLM")
                
                # Check if response is too short or nonsensical (common with GPT-2)
                if len(answer.strip()) < 10 or "I don't" in answer.lower() or "cannot" in answer.lower():
                    logger.warning("LLM returned very short or unclear response, falling back to document search")
                    raise ValueError("Unclear response from LLM")
                    
            except Exception as generation_error:
                logger.error(f"Generation error: {str(generation_error)}")
                # Try a simpler approach - just search documents
                docs = self.vector_store.similarity_search(question, k=3)
                if docs:
                    # Create a simple answer from retrieved documents
                    context = "\n".join([doc.page_content for doc in docs])
                    
                    # Create a more structured answer
                    answer = f"Based on your documents, here's what I found:\n\n"
                    for i, doc in enumerate(docs, 1):
                        source_name = doc.metadata.get("source", "Unknown")
                        content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                        answer += f"**Document {i} ({source_name}):**\n{content}\n\n"
                    
                    # Add a helpful note
                    answer += "💡 *Note: This information was retrieved from your documents. For more specific answers, try asking more detailed questions about the content.*"
                    
                    sources = [{"source": doc.metadata.get("source", "Unknown"), 
                               "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content} 
                              for doc in docs]
                    return {
                        "answer": answer,
                        "sources": sources,
                        "question": question
                    }
                else:
                    return {
                        "answer": "I couldn't find any relevant information in the documents to answer your question. Please try rephrasing your question or check if the documents have been processed correctly.",
                        "sources": [],
                        "question": question
                    }
            
            # Extract sources
            sources = []
            if response.get("source_documents"):
                for doc in response["source_documents"]:
                    if doc.metadata.get("source"):
                        sources.append({
                            "source": doc.metadata["source"],
                            "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        })
            
            return {
                "answer": answer,
                "sources": sources,
                "question": question
            }
            
        except Exception as e:
            logger.error(f"Error asking question: {str(e)}")
            return {
                "answer": f"Sorry, I encountered an error while processing your question. Please try rephrasing it or ask a different question.",
                "sources": [],
                "question": question
            }
    
    def get_document_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about a document without processing it"""
        return self.document_processor.get_document_info(file_path)
    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return self.vector_store.get_collection_stats()
    
    def get_document_sources(self) -> List[str]:
        """Get list of document sources in the vector store"""
        return self.vector_store.get_document_sources()
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")
    
    def clear_vector_store(self):
        """Clear all documents from vector store"""
        self.vector_store.clear_all()
        logger.info("Vector store cleared")
    
    def search_documents(self, query: str, k: int = 4) -> List[Document]:
        """Search documents without generating an answer"""
        return self.vector_store.similarity_search(query, k=k)
    
    def search_documents_with_scores(self, query: str, k: int = 4) -> List[tuple]:
        """Search documents with similarity scores"""
        return self.vector_store.similarity_search_with_score(query, k=k)