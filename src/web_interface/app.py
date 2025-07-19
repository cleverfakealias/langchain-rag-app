import streamlit as st
import os
import tempfile
from typing import List, Dict, Any
import time

from ..rag_engine import RAGEngine
from ..utils import load_environment, create_directories, is_supported_file, format_file_size, sanitize_filename

# Page configuration
st.set_page_config(
    page_title="LangChain RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-box {
        background-color: #fff3e0;
        border: 1px solid #ffcc02;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_engine():
    """Initialize RAG engine with caching"""
    try:
        # Load environment and create directories
        env_vars = load_environment()
        create_directories()
        
        # Initialize RAG engine
        rag_engine = RAGEngine(env_vars["openai_api_key"])
        return rag_engine
    except Exception as e:
        st.error(f"Failed to initialize RAG engine: {str(e)}")
        return None

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to documents directory"""
    try:
        # Create documents directory if it doesn't exist
        os.makedirs("documents", exist_ok=True)
        
        # Sanitize filename
        filename = sanitize_filename(uploaded_file.name)
        file_path = os.path.join("documents", filename)
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def display_chat_message(message: str, is_user: bool = True, sources: List[Dict] = None):
    """Display a chat message with proper styling"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>Assistant:</strong> {message}
        </div>
        """, unsafe_allow_html=True)
        
        # Display sources if available
        if sources:
            st.markdown("**Sources:**")
            for source in sources:
                st.markdown(f"""
                <div class="source-box">
                    <strong>File:</strong> {source['source']}<br>
                    <strong>Content:</strong> {source['content']}
                </div>
                """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 LangChain RAG Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize RAG engine
    rag_engine = initialize_rag_engine()
    if not rag_engine:
        st.error("Failed to initialize RAG engine. Please check your OpenAI API key.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            help="Upload PDF, TXT, MD, or DOCX files"
        )
        
        if uploaded_files:
            st.write(f"**Uploaded Files:** {len(uploaded_files)}")
            for file in uploaded_files:
                st.write(f"• {file.name} ({format_file_size(file.size)})")
        
        # Process documents button
        if st.button("🔄 Process Documents", type="primary"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    # Save uploaded files
                    file_paths = []
                    for uploaded_file in uploaded_files:
                        file_path = save_uploaded_file(uploaded_file)
                        if file_path:
                            file_paths.append(file_path)
                    
                    if file_paths:
                        # Process documents
                        results = rag_engine.process_documents(file_paths)
                        
                        # Display results
                        if results["success"]:
                            st.success(f"✅ Successfully processed {results['total_processed']} documents ({results['total_chunks']} chunks)")
                            for success in results["success"]:
                                st.write(f"• {success['filename']} ({success['chunks']} chunks)")
                        
                        if results["errors"]:
                            st.error(f"❌ {len(results['errors'])} documents failed to process")
                            for error in results["errors"]:
                                st.write(f"• {os.path.basename(error['file_path'])}: {error['error']}")
            else:
                st.warning("Please upload documents first")
        
        # Vector store stats
        st.markdown("---")
        st.markdown("## 📊 Vector Store Stats")
        stats = rag_engine.get_vector_store_stats()
        if "error" not in stats:
            st.write(f"**Total Documents:** {stats['total_documents']}")
            st.write(f"**Embedding Model:** {stats['embedding_model']}")
        else:
            st.write("No documents in vector store")
        
        # Document sources
        sources = rag_engine.get_document_sources()
        if sources:
            st.markdown("**Document Sources:**")
            for source in sources:
                st.write(f"• {source}")
        
        # Clear options
        st.markdown("---")
        st.markdown("## 🗑️ Clear Data")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat"):
                rag_engine.clear_memory()
                st.success("Chat history cleared!")
        with col2:
            if st.button("Clear Documents"):
                rag_engine.clear_vector_store()
                st.success("Vector store cleared!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">💬 Chat with Your Documents</h2>', unsafe_allow_html=True)
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            display_chat_message(
                message["content"],
                message["is_user"],
                message.get("sources")
            )
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"content": prompt, "is_user": True})
            display_chat_message(prompt, is_user=True)
            
            # Get response from RAG engine
            with st.spinner("Thinking..."):
                response = rag_engine.ask_question(prompt)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "content": response["answer"],
                    "is_user": False,
                    "sources": response["sources"]
                })
                
                # Display response
                display_chat_message(
                    response["answer"],
                    is_user=False,
                    sources=response["sources"]
                )
    
    with col2:
        st.markdown('<h2 class="sub-header">🔍 Quick Search</h2>', unsafe_allow_html=True)
        
        # Search input
        search_query = st.text_input("Search documents...")
        if search_query and st.button("Search"):
            with st.spinner("Searching..."):
                search_results = rag_engine.search_documents_with_scores(search_query, k=3)
                
                if search_results:
                    st.markdown("**Search Results:**")
                    for doc, score in search_results:
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source:</strong> {doc.metadata.get('source', 'Unknown')}<br>
                            <strong>Score:</strong> {score:.3f}<br>
                            <strong>Content:</strong> {doc.page_content[:150]}...
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No relevant documents found")

if __name__ == "__main__":
    main() 