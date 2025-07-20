import streamlit as st
from typing import List, Dict, Optional

def render_header():
    st.markdown('<h1 class="main-header">🤖 LangChain RAG Assistant</h1>', unsafe_allow_html=True)

def render_sidebar_uploaded_files(uploaded_files, format_file_size):
    if uploaded_files:
        st.write(f"**Uploaded Files:** {len(uploaded_files)}")
        for file in uploaded_files:
            st.write(f"• {file.name} ({format_file_size(file.size)})")

def render_sidebar_status(llm_config, embedding_config, devices):
    st.markdown("---")
    st.markdown("## ⚙️ System Status")
    st.write(f"**🤖 LLM Model:** {llm_config.name}")
    st.write(f"**🔤 Embedding Model:** {embedding_config.name}")
    st.write(f"**💻 Device:** {devices['recommended_device'].upper()}")
    if devices['cuda']:
        st.write(f"**🎮 GPU:** {devices['cuda_name']}")
    st.write(f"**🌡️ Temperature:** {llm_config.temperature}")
    st.write(f"**📝 Max Tokens:** {llm_config.max_new_tokens}")
    if devices['cuda']:
        st.success("✅ GPU acceleration active")
    else:
        st.info("ℹ️ Running on CPU")

def render_vector_store_stats(stats):
    st.markdown("---")
    st.markdown("## 📊 Vector Store Stats")
    if "error" not in stats:
        st.write(f"**Total Documents:** {stats['total_documents']}")
        st.write(f"**Embedding Model:** {stats['embedding_model']}")
    else:
        st.write("No documents in vector store")

def render_document_sources(sources):
    if sources:
        st.markdown("**Document Sources:**")
        for source in sources:
            st.write(f"• {source}")

def render_clear_data_buttons(rag_engine):
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

def render_chat_history(messages, display_chat_message):
    for message in messages:
        display_chat_message(
            message["content"],
            message["is_user"],
            message.get("sources")
        )

def render_quick_search(rag_engine):
    st.markdown('<h2 class="sub-header">🔍 Quick Search</h2>', unsafe_allow_html=True)
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
                    <div class=\"source-box\">
                        <strong>Result #{i}:</strong> {doc.metadata.get('source', 'Unknown')}<br>
                        <strong>Relevance Score:</strong> {score:.3f} ({(score*100):.1f}%)<br>
                        <strong>Content:</strong> {doc.page_content[:150]}...
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("❌ No relevant documents found")
                status.update(label="❌ No results found", state="error")
                st.info("Try using different keywords or check if documents are processed")
