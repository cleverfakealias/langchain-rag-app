import os
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import logging

from ..document_processor.processor import DocumentProcessor
from ..vector_store.store import VectorStore

logger = logging.getLogger(__name__)

class RAGEngine:
    """Main RAG engine that orchestrates document processing, retrieval, and generation"""
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name=model_name,
            temperature=0.7
        )
        
        # Initialize conversation chain
        self.conversation_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self._setup_conversation_chain()
    
    def _setup_conversation_chain(self):
        """Setup the conversational retrieval chain"""
        try:
            # Custom prompt template
            prompt_template = """You are a helpful AI assistant that answers questions based on the provided context. 
            Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create conversation chain
            self.conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vector_store.vector_store.as_retriever(
                    search_kwargs={"k": 4}
                ),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                return_source_documents=True,
                verbose=False
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
            
            # Get response
            response = self.conversation_chain({"question": question})
            
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
                "answer": response["answer"],
                "sources": sources,
                "question": question
            }
            
        except Exception as e:
            logger.error(f"Error asking question: {str(e)}")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
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