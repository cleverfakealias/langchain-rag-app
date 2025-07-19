#!/usr/bin/env python3
"""
Simple test script for the RAG system
"""

import os
import sys
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_engine import RAGEngine
from src.utils import load_environment, create_directories

def create_test_document():
    """Create a simple test document"""
    test_content = """
    LangChain is a framework for developing applications powered by language models.
    
    Key Features:
    1. Modular Components: LangChain provides modular components for working with language models.
    2. Prompt Management: Tools for managing and optimizing prompts.
    3. Memory: Built-in memory systems for maintaining conversation context.
    4. Document Loaders: Support for loading documents from various sources.
    5. Vector Stores: Integration with vector databases for similarity search.
    6. Chains: Combine multiple components into complex workflows.
    7. Agents: Build autonomous agents that can use tools and reason.
    
    RAG (Retrieval-Augmented Generation) is a technique that combines:
    - Document retrieval: Finding relevant documents based on a query
    - Text generation: Using a language model to generate answers based on retrieved context
    
    This approach helps language models provide more accurate and up-to-date information
    by grounding their responses in specific documents rather than relying solely on
    their training data.
    """
    
    # Create test document
    os.makedirs("documents", exist_ok=True)
    test_file_path = "documents/test_langchain.txt"
    
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    return test_file_path

def test_rag_system():
    """Test the RAG system with a sample document"""
    print("🤖 Testing LangChain RAG System")
    print("=" * 50)
    
    try:
        # Load environment
        print("1. Loading environment...")
        env_vars = load_environment()
        create_directories()
        print("✅ Environment loaded successfully")
        
        # Initialize RAG engine
        print("2. Initializing RAG engine...")
        rag_engine = RAGEngine(env_vars["openai_api_key"])
        print("✅ RAG engine initialized successfully")
        
        # Create test document
        print("3. Creating test document...")
        test_file = create_test_document()
        print(f"✅ Test document created: {test_file}")
        
        # Process document
        print("4. Processing test document...")
        results = rag_engine.process_documents([test_file])
        print(f"✅ Document processed: {results['total_processed']} files, {results['total_chunks']} chunks")
        
        # Test questions
        test_questions = [
            "What is LangChain?",
            "What are the key features of LangChain?",
            "What is RAG and how does it work?",
            "How does document retrieval work in RAG?"
        ]
        
        print("5. Testing questions...")
        for i, question in enumerate(test_questions, 1):
            print(f"\nQ{i}: {question}")
            response = rag_engine.ask_question(question)
            print(f"A{i}: {response['answer'][:200]}...")
            if response['sources']:
                print(f"   Sources: {len(response['sources'])} documents")
        
        # Test vector store stats
        print("\n6. Checking vector store stats...")
        stats = rag_engine.get_vector_store_stats()
        print(f"✅ Vector store contains {stats['total_documents']} documents")
        
        print("\n🎉 All tests passed! RAG system is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_rag_system()
    sys.exit(0 if success else 1) 