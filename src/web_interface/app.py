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

def show_error_modal(title: str, error_message: str, details: str = None):
    """Display a formatted error modal"""
    st.markdown(f"""
    <div style="
        background-color: #ffebee;
        border: 2px solid #f44336;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #b71c1c;
    ">
        <h4 style="margin: 0 0 0.5rem 0; color: #d32f2f;">❌ {title}</h4>
        <p style="margin: 0.5rem 0; font-weight: 500;">{error_message}</p>
        {f'<details style="margin-top: 0.5rem;"><summary style="cursor: pointer; color: #d32f2f;">🔍 Show Details</summary><pre style="background: #fff3e0; padding: 0.5rem; border-radius: 4px; margin-top: 0.5rem; font-size: 12px; overflow-x: auto;">{details}</pre></details>' if details else ''}
    </div>
    """, unsafe_allow_html=True)

def show_success_message(message: str):
    """Display a formatted success message"""
    st.markdown(f"""
    <div style="
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #1b5e20;
    ">
        <h4 style="margin: 0 0 0.5rem 0; color: #2e7d32;">✅ Success</h4>
        <p style="margin: 0.5rem 0; font-weight: 500;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

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

# Custom CSS to make sidebar wider
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        min-width: 400px !important;
        max-width: 500px !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        width: 100% !important;
    }
    
    /* Make tabs more compact */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        padding: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 12px;
        padding: 6px 8px;
        min-width: 60px;
    }
    
    /* Improve sidebar content spacing */
    .css-1d391kg {
        padding: 1rem;
    }
    
    /* Make buttons more compact */
    .stButton > button {
        font-size: 14px;
        padding: 0.4rem 0.8rem;
    }
    
    /* Compact metrics */
    .stMetric {
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

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
    
    # Sidebar with organized tabs
    with st.sidebar:
        st.markdown("## ⚙️ Settings & Management")
        
        # Get document sources once for all tabs
        sources = rag_engine.get_document_sources()
        
        # Create tabs for better organization
        tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload", "⚙️ Settings", "📊 Stats", "🗑️ Cleanup"])
        
        # Tab 1: Document Upload
        with tab1:
            st.markdown("### 📁 Upload Documents")
            
            uploaded_files = st.file_uploader(
                "Choose files",
                type=["pdf", "txt", "md", "docx"],
                accept_multiple_files=True,
                help="Upload PDF, TXT, MD, or DOCX files"
            )
            
            if uploaded_files:
                st.write(f"**Files to upload:** {len(uploaded_files)}")
                for file in uploaded_files:
                    st.write(f"• {file.name} ({format_file_size(file.size)})")
            
            # Process documents button
            if st.button("🔄 Process Documents", type="primary", use_container_width=True):
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
                                
                                if results["errors"]:
                                    st.write("❌ Errors occurred:")
                                    for error in results["errors"]:
                                        if isinstance(error, dict):
                                            error_msg = error.get('error', str(error))
                                            file_path = error.get('file_path', 'Unknown file')
                                            show_error_modal(
                                                f"Processing Error: {os.path.basename(file_path)}",
                                                f"Failed to process {os.path.basename(file_path)}",
                                                error_msg
                                            )
                                        else:
                                            show_error_modal(
                                                "Processing Error",
                                                "An error occurred during document processing",
                                                str(error)
                                            )
                                
                                status.update(label="✅ Processing complete!", state="complete")
                                st.rerun()
                else:
                    st.warning("Please upload files first")
        
        # Tab 2: Settings & Configuration
        with tab2:
            st.markdown("### ⚙️ Retrieval Settings")
            
            # Chunking settings
            st.markdown("**📄 Chunking**")
            chunk_size = st.slider(
                "Chunk Size",
                min_value=100,
                max_value=2000,
                value=1000,
                step=100,
                help="Size of text chunks for processing"
            )
            
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=200,
                step=50,
                help="Overlap between consecutive chunks"
            )
            
            # Retrieval method
            st.markdown("**🔍 Retrieval Method**")
            retrieval_method = st.selectbox(
                "Method",
                ["mmr", "similarity", "hybrid"],
                index=0,
                help="MMR: Better diversity, Similarity: Standard search, Hybrid: Combines semantic and keyword search"
            )
            
            if retrieval_method == "mmr":
                mmr_lambda = st.slider(
                    "MMR Diversity (λ)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.1,
                    help="0 = Maximum diversity, 1 = Maximum relevance"
                )
                fetch_k = st.slider(
                    "Fetch K (before MMR)",
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=5,
                    help="Number of documents to fetch before applying MMR selection"
                )
            elif retrieval_method == "hybrid":
                hybrid_alpha = st.slider(
                    "Hybrid Weight (α)",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="0 = Keyword only, 1 = Semantic only"
                )
            
            # Update settings button
            if st.button("🔄 Update Settings", use_container_width=True):
                # Update chunking settings
                rag_engine.set_chunking_config(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                
                # Update retrieval settings
                if retrieval_method == "mmr":
                    rag_engine.set_retrieval_config(
                        method=retrieval_method,
                        mmr_lambda=mmr_lambda,
                        fetch_k=fetch_k
                    )
                elif retrieval_method == "hybrid":
                    rag_engine.set_retrieval_config(
                        method=retrieval_method,
                        hybrid_alpha=hybrid_alpha
                    )
                else:
                    rag_engine.set_retrieval_config(method=retrieval_method)
                
                show_success_message("Settings updated successfully!")
        
        # Tab 3: Statistics & Monitoring
        with tab3:
            st.markdown("### 📊 System Statistics")
            
            # Vector store stats
            stats = rag_engine.get_vector_store_stats()
            if "error" not in stats:
                st.metric("Total Documents", stats['total_documents'])
                st.write(f"**Embedding Model:** {stats['embedding_model']}")
            else:
                st.info("No documents in vector store")
            
            # Enhanced Document Statistics
            doc_stats = rag_engine.get_document_statistics()
            if "error" not in doc_stats:
                summary = doc_stats['summary']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Indexed Files", summary['indexed_files'])
                with col2:
                    st.metric("Unindexed Files", summary['unindexed_files'])
                
                if summary['orphaned_vectors'] > 0:
                    st.warning(f"⚠️ {summary['orphaned_vectors']} orphaned vectors")
                    
                    # Cleanup orphaned vectors
                    if st.button("🧹 Cleanup Orphaned", use_container_width=True):
                        try:
                            # Get orphaned sources
                            vector_sources = doc_stats['vector_store']['sources']
                            physical_files = [doc['name'] for doc in doc_stats['physical_files']['files']]
                            orphaned_sources = [s for s in vector_sources if s not in physical_files]
                            
                            # Delete orphaned vectors
                            for source in orphaned_sources:
                                rag_engine.delete_document_from_vector_store(source)
                            
                            show_success_message(f"Successfully cleaned up {len(orphaned_sources)} orphaned vectors")
                            st.rerun()
                        except Exception as e:
                            show_error_modal(
                                "Cleanup Error",
                                "Failed to cleanup orphaned vectors",
                                str(e)
                            )
            
            # Document sources (collapsible)
            if sources:
                with st.expander("📋 Document Sources", expanded=False):
                    for source in sources:
                        st.write(f"• {source}")
        
        # Tab 4: Document Management & Cleanup
        with tab4:
            st.markdown("### 📁 Document Management")
            
            # Get list of physical documents
            documents_dir = "documents"
            if os.path.exists(documents_dir):
                physical_docs = []
                for file in os.listdir(documents_dir):
                    if file.lower().endswith(('.pdf', '.txt', '.md', '.docx')):
                        file_path = os.path.join(documents_dir, file)
                        file_size = os.path.getsize(file_path)
                        physical_docs.append({
                            'name': file,
                            'path': file_path,
                            'size': file_size
                        })
                
                if physical_docs:
                    st.write(f"**Documents ({len(physical_docs)}):**")
                    
                    # Create a compact list with delete buttons
                    for i, doc in enumerate(physical_docs):
                        col1, col2, col3 = st.columns([4, 1, 1])
                        
                        with col1:
                            st.write(f"📄 {doc['name']}")
                            st.caption(f"{format_file_size(doc['size'])}")
                        
                        with col2:
                            # Check if document is in vector store
                            in_vector_store = doc['name'] in sources if sources else False
                            if in_vector_store:
                                st.success("✅")
                            else:
                                st.warning("⚠️")
                        
                        with col3:
                            # Delete button for each document
                            if st.button(f"🗑️", key=f"delete_{i}", help=f"Delete {doc['name']}"):
                                try:
                                    # Delete from vector store first
                                    rag_engine.delete_document_from_vector_store(doc['name'])
                                    
                                    # Delete physical file
                                    os.remove(doc['path'])
                                    
                                    show_success_message(f"Successfully deleted {doc['name']}")
                                    st.rerun()  # Refresh the page
                                except Exception as e:
                                    show_error_modal(
                                        f"Delete Error: {doc['name']}",
                                        f"Failed to delete {doc['name']}",
                                        str(e)
                                    )
                else:
                    st.info("No physical documents found")
            else:
                st.info("Documents directory not found")
            
            # Bulk operations
            st.markdown("---")
            st.markdown("### 🗑️ Bulk Operations")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Clear Chat", use_container_width=True):
                    rag_engine.clear_memory()
                    show_success_message("Chat history cleared successfully!")
            
            with col2:
                if st.button("Clear Vectors", use_container_width=True):
                    rag_engine.clear_vector_store()
                    show_success_message("Vector store cleared successfully!")
            
            # Clear all documents (with confirmation)
            if st.button("🗑️ Clear All Documents", type="secondary", use_container_width=True):
                # Confirmation dialog
                if st.session_state.get('confirm_delete_all', False):
                    try:
                        # Clear vector store
                        rag_engine.clear_vector_store()
                        
                        # Delete all physical documents
                        documents_dir = "documents"
                        if os.path.exists(documents_dir):
                            deleted_count = 0
                            for file in os.listdir(documents_dir):
                                if file.lower().endswith(('.pdf', '.txt', '.md', '.docx')):
                                    file_path = os.path.join(documents_dir, file)
                                    os.remove(file_path)
                                    deleted_count += 1
                            
                            show_success_message(f"Successfully deleted {deleted_count} documents and cleared vector store")
                        else:
                            show_success_message("Vector store cleared successfully")
                        
                        # Reset confirmation state
                        st.session_state.confirm_delete_all = False
                        st.rerun()
                    except Exception as e:
                        show_error_modal(
                            "Clear All Error",
                            "Failed to clear all documents",
                            str(e)
                        )
                else:
                    st.session_state.confirm_delete_all = True
                    st.warning("⚠️ Click again to confirm deletion of ALL documents")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
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
            
            # Display user message
            display_chat_message(prompt, True)
            
            # Get AI response
            with st.spinner("🤖 Thinking..."):
                try:
                    response = rag_engine.ask_question(prompt)
                    
                    # Add AI response to chat history
                    st.session_state.messages.append({
                        "content": response["answer"],
                        "is_user": False,
                        "sources": response.get("sources", [])
                    })
                    
                    # Display AI response
                    display_chat_message(
                        response["answer"],
                        False,
                        response.get("sources", [])
                    )
                    
                except Exception as e:
                    error_message = f"Failed to get response from AI"
                    show_error_modal(
                        "Chat Error",
                        error_message,
                        str(e)
                    )
                    st.session_state.messages.append({"content": error_message, "is_user": False})
    
    with col2:
        # System Status Panel
        st.markdown("### 🖥️ System Status")
        
        try:
            from config.config import Config
            config = Config()
            llm_config = config.get_current_llm_config()
            embedding_config = config.get_current_embedding_config()
            devices = config.detect_available_devices()
            
            # Model information
            st.markdown("**🤖 Models**")
            st.write(f"LLM: {llm_config.name}")
            st.write(f"Embedding: {embedding_config.name}")
            
            # Device information
            st.markdown("**💻 Hardware**")
            st.write(f"Device: {devices['recommended_device'].upper()}")
            if devices['cuda']:
                st.success(f"GPU: {devices['cuda_name']}")
            else:
                st.info("CPU Mode")
            
            # Configuration
            st.markdown("**⚙️ Settings**")
            st.write(f"Temperature: {llm_config.temperature}")
            st.write(f"Max Tokens: {llm_config.max_new_tokens}")
            
            # Quick stats
            doc_stats = rag_engine.get_document_statistics()
            if "error" not in doc_stats:
                summary = doc_stats['summary']
                st.markdown("**📊 Quick Stats**")
                st.metric("Documents", summary['indexed_files'])
                st.metric("Chunks", doc_stats['vector_store']['total_documents'])
                
        except Exception as e:
            st.warning(f"⚠️ Status unavailable: {str(e)}")
        
        # Quick actions
        st.markdown("### ⚡ Quick Actions")
        
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            rag_engine.clear_memory()
            st.session_state.messages = []
            show_success_message("Chat cleared successfully!")
            st.rerun()

if __name__ == "__main__":
    main() 