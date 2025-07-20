import streamlit as st
import os
import sys
from typing import List, Dict, Any, Optional

from src.web_interface.file_utils import save_uploaded_file
from src.web_interface.chat_utils import display_chat_message
from src.web_interface.engine_init import initialize_rag_engine

# Add the src directory to the Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_engine import RAGEngine
from utils import create_directories, is_supported_file, format_file_size, sanitize_filename

# Page configuration
st.set_page_config(
    page_title="LangChain RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/langchain-rag-app',
        'Report a bug': 'https://github.com/your-repo/langchain-rag-app/issues',
        'About': 'LangChain RAG Assistant - A powerful document Q&A system'
    }
)


# Load custom CSS from styles.css
css_path = os.path.join(os.path.dirname(__file__), "styles.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)





def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 LangChain RAG Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize RAG engine
    rag_engine = initialize_rag_engine()
    if not rag_engine:
        st.error("Failed to initialize RAG engine. Please check the console for error details.")
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
                # Create progress container
                progress_container = st.container()
                
                with progress_container:
                    with st.status("📄 Processing documents...", expanded=True) as status:
                        # Step 1: Save files
                        st.write("💾 Saving uploaded files...")
                        file_paths = []
                        for i, uploaded_file in enumerate(uploaded_files):
                            st.write(f"  • Saving {uploaded_file.name}...")
                            file_path = save_uploaded_file(uploaded_file)
                            if file_path:
                                file_paths.append(file_path)
                                st.write(f"    ✅ Saved successfully")
                            else:
                                st.write(f"    ❌ Failed to save")
                        
                        if file_paths:
                            # Step 2: Process documents
                            st.write("🔧 Processing documents with AI...")
                            st.write("  • Loading language models...")
                            st.write("  • Extracting text and creating embeddings...")
                            st.write("  • Storing in vector database...")
                            
                            results = rag_engine.process_documents(file_paths)
                            
                            # Step 3: Display results
                            if results["success"]:
                                st.write(f"✅ Successfully processed {results['total_processed']} documents")
                                st.write(f"📊 Created {results['total_chunks']} text chunks")
                                for success in results["success"]:
                                    st.write(f"  • {success['filename']} ({success['chunks']} chunks)")
                                
                                status.update(label="🎉 Documents processed successfully!", state="complete")
                            else:
                                status.update(label="⚠️ Some documents failed to process", state="error")
                            
                            if results["errors"]:
                                st.write(f"❌ {len(results['errors'])} documents failed:")
                                for error in results["errors"]:
                                    st.write(f"  • {os.path.basename(error['file_path'])}: {error['error']}")
                        else:
                            status.update(label="❌ No files were saved successfully", state="error")
            else:
                st.warning("Please upload documents first")
        
        # System Status
        st.markdown("---")
        st.markdown("## ⚙️ System Status")
        
        # Get current configuration
        try:
            from config.config import Config
            config = Config()
            llm_config = config.get_current_llm_config()
            embedding_config = config.get_current_embedding_config()
            devices = config.detect_available_devices()
            
            st.write(f"**🤖 LLM Model:** {llm_config.name}")
            st.write(f"**🔤 Embedding Model:** {embedding_config.name}")
            st.write(f"**💻 Device:** {devices['recommended_device'].upper()}")
            if devices['cuda']:
                st.write(f"**🎮 GPU:** {devices['cuda_name']}")
            st.write(f"**🌡️ Temperature:** {llm_config.temperature}")
            st.write(f"**📝 Max Tokens:** {llm_config.max_new_tokens}")
            
            # Show device status with color
            if devices['cuda']:
                st.success("✅ GPU acceleration active")
            else:
                st.info("ℹ️ Running on CPU")
                
        except Exception as e:
            st.warning(f"⚠️ Could not load system status: {str(e)}")
        
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
            
            # Create a container for progress updates
            progress_container = st.container()
            
            # Get response from RAG engine with detailed progress
            with progress_container:
                # Step 1: Searching documents
                with st.status("🔍 Searching documents...", expanded=True) as status:
                    st.write("Looking for relevant information in your documents...")
                    response = rag_engine.ask_question(prompt)
                    status.update(label="✅ Search completed!", state="complete")
                
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
            with st.status("🔍 Searching documents...", expanded=True) as status:
                st.write(f"Looking for documents matching: '{search_query}'")
                st.write("• Searching vector database...")
                st.write("• Calculating similarity scores...")
                
                search_results = rag_engine.search_documents_with_scores(search_query, k=3)
                
                if search_results:
                    st.write(f"✅ Found {len(search_results)} relevant documents")
                    status.update(label="✅ Search completed!", state="complete")
                    
                    st.markdown("**Search Results:**")
                    for i, (doc, score) in enumerate(search_results, 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Result #{i}:</strong> {doc.metadata.get('source', 'Unknown')}<br>
                            <strong>Relevance Score:</strong> {score:.3f} ({(score*100):.1f}%)<br>
                            <strong>Content:</strong> {doc.page_content[:150]}...
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.write("❌ No relevant documents found")
                    status.update(label="❌ No results found", state="error")
                    st.info("Try using different keywords or check if documents are processed")

if __name__ == "__main__":
    main() 